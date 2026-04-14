import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, einsum
from torch import Tensor

from layers.Embed import DataEmbedding

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_inner, dt_rank, d_conv=4, d_ff=None):
        super(MambaBlock, self).__init__()
        self.d_inner = d_inner
        self.dt_rank = dt_rank
        if d_ff is None:
            d_ff = 4 * d_model

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )

        # takes in x and outputs the input-specific delta, B, C
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_ff * 2, bias=False)

        # projects delta
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = repeat(torch.arange(1, d_ff + 1), "n -> d n", d=self.d_inner)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        Figure 3 in Section 3.4 in the paper
        """
        (b, l, d) = x.shape

        x_and_res = self.in_proj(x)  # [B, L, 2 * d_inner]
        (x, res) = x_and_res.split(split_size=[self.d_inner, self.d_inner], dim=-1)

        x = rearrange(x, "b l d -> b d l")
        x = self.conv1d(x)[:, :, :l]
        x = rearrange(x, "b d l -> b l d")

        x = F.silu(x)

        y = self.ssm(x)
        y = y * F.silu(res)

        output = self.out_proj(y)
        return output

    def ssm(self, x):
        """
        Algorithm 2 in Section 3.2 in the paper
        """
        (d_in, n) = self.A_log.shape

        A = -torch.exp(self.A_log.float())  # [d_in, n]
        D = self.D.float()  # [d_in]

        x_dbl = self.x_proj(x)  # [B, L, d_rank + 2 * d_ff]
        (delta, B, C) = x_dbl.split(split_size=[self.dt_rank, n, n], dim=-1)  # delta: [B, L, d_rank]; B, C: [B, L, n]
        delta = F.softplus(self.dt_proj(delta))  # [B, L, d_in]
        y = self.selective_scan(x, delta, A, B, C, D)

        return y

    def selective_scan(self, u, delta, A, B, C, D):
        (b, l, d_in) = u.shape
        n = A.shape[1]

        deltaA = torch.exp(
            einsum(delta, A, "b l d, d n -> b l d n"))  # A is discretized using ZOH discretization
        deltaB_u = einsum(delta, B, u,
                      "b l d, b l n, b l d -> b l d n")  # B is discretized using simplified Euler discretization

        # selective scan, sequential instead of parallel
        x = torch.zeros((b, d_in, n), device=deltaA.device)
        ys = []
        for i in range(l):
            x = deltaA[:, i] * x + deltaB_u[:, i]
            y = einsum(x, C[:, i, :], "b d n, b n -> b d")
            ys.append(y)

        y = torch.stack(ys, dim=1)  # [B, L, d_in]
        y = y + u * D

        return y

class ResidualBlock(nn.Module):
    def __init__(self, d_model, d_inner, dt_rank, d_conv=4, d_ff=None):
        super(ResidualBlock, self).__init__()

        self.mixer = MambaBlock(d_model, d_inner, dt_rank, d_conv, d_ff)
        self.norm = RMSNorm(d_model)

    def forward(self, x):
        output = self.mixer(self.norm(x)) + x
        return output

class TSMambaEncoder(nn.Module):
    """Time series Mamba encoder module.

    Args:
        feat_dim: feature dimension
        max_len: maximum length of the input sequence
        d_model: the embed dim
        num_layers: the number of mamba layers
        dropout: the dropout value
        activation: the activation function of intermediate layer, relu or gelu
        expand: expansion factor for inner dimension
    """

    def __init__(
        self,
        feat_dim: int,
        max_len: int,
        d_model: int,
        num_layers: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        expand: int = 2,
    ):
        super(TSMambaEncoder, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        
        # Project input to d_model dimension
        self.project_inp = nn.Linear(feat_dim, d_model)
        
        # Positional encoding is inherently handled in Mamba through the SSM mechanism
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, self.d_inner, self.dt_rank) 
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        self.output_layer = nn.Linear(d_model, feat_dim)
        
        # Activation and dropout
        self.act = F.gelu if activation == "gelu" else F.relu
        self.dropout = nn.Dropout(dropout)
        
        self.feat_dim = feat_dim

    def forward(self, X: Tensor, padding_masks: Tensor) -> Tensor:
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        # Project input to d_model dimension
        inp = self.project_inp(X) * math.sqrt(self.d_model)
        
        # Apply padding mask
        inp = inp * padding_masks.unsqueeze(-1)
        
        # Pass through Mamba layers
        output = inp
        for layer in self.layers:
            output = layer(output)
            
        output = self.norm(output)
        output = self.act(output)
        output = self.dropout(output)
        
        # Project back to feature dimension
        output = self.output_layer(output)
        
        # Apply padding mask again to ensure padding is zeroed out
        output = output * padding_masks.unsqueeze(-1)
        
        return output

class TSMambaClassiregressor(nn.Module):
    """
    Mamba-based classifier/regressor for time series data.
    
    Args:
        feat_dim: feature dimension
        max_len: maximum length of the input sequence
        d_model: the embed dim
        num_layers: the number of mamba layers
        num_classes: the number of classes in the classification task
        dropout: the dropout value
        activation: the activation function of intermediate layer, relu or gelu
        expand: expansion factor for inner dimension
    """

    def __init__(
        self,
        feat_dim: int,
        max_len: int,
        d_model: int,
        num_layers: int,
        num_classes: int,
        dropout: float = 0.1,
        activation: str = "gelu",
        expand: int=4,
    ):
        super(TSMambaClassiregressor, self).__init__()

        self.max_len = max_len
        self.d_model = d_model
        
        # Project input to d_model dimension
        self.project_inp = nn.Linear(feat_dim, d_model)
        
        # Mamba configuration
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)
        
        # Mamba layers
        self.layers = nn.ModuleList([
            ResidualBlock(d_model, self.d_inner, self.dt_rank) 
            for _ in range(num_layers)
        ])
        
        self.norm = RMSNorm(d_model)
        
        # Activation and dropout
        self.act = F.gelu if activation == "gelu" else F.relu
        self.dropout = nn.Dropout(dropout)
        
        # Classification/regression head
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        self.output_layer = self.build_output_module(d_model, max_len, num_classes)

    def build_output_module(
        self, d_model: int, max_len: int, num_classes: int
    ) -> nn.Module:
        """ Build linear layer that maps from d_model*max_len to num_classes.

        Args:
            d_model: the embed dim
            max_len: maximum length of the input sequence
            num_classes: the number of classes in the classification task

        Returns:
            output_layer: Module that outputs tensor of shape (batch_size, num_classes)
        """
        output_layer = nn.Linear(d_model * max_len, num_classes)
        return output_layer

    def forward(self, X: Tensor, padding_masks: Tensor = None) -> Tensor:
        """
        Args:
            X: (batch_size, seq_length, feat_dim) torch tensor of masked features (input)
            padding_masks: (batch_size, seq_length) boolean tensor, 1 means keep vector at this position, 0 means padding
        Returns:
            output: (batch_size, num_classes)
        """
        if padding_masks is None:
            padding_masks = torch.ones(X.shape[0], X.shape[1], dtype=torch.bool, device=X.device)

        # Project input to d_model dimension
        inp = self.project_inp(X) * math.sqrt(self.d_model)
        
        # Apply padding mask
        inp = inp * padding_masks.unsqueeze(-1)
        
        # Pass through Mamba layers
        output = inp
        for layer in self.layers:
            output = layer(output)
            
        output = self.norm(output)
        output = self.act(output)
        output = self.dropout(output)
        
        # Apply padding mask to ensure padding is zeroed out
        output = output * padding_masks.unsqueeze(-1)
        
        # Flatten and project to output classes
        output = output.reshape(output.shape[0], -1)  # (batch_size, seq_length * d_model)
        output = self.output_layer(output)  # (batch_size, num_classes)
        
        return output

# Test code
if __name__ == "__main__":
    print("Testing TSMambaClassiregressor...")
    x = torch.rand(2, 3, 4)  # (batch_size, seq_length, feat_dim)
    padding_masks = torch.tensor([[True, True, True], [True, True, True]])
    model = TSMambaClassiregressor(
        feat_dim=4,
        max_len=x.shape[1],
        d_model=4,
        num_layers=2,
        num_classes=1,
        dropout=0.1,
        activation="gelu",
        expand=2,
    )
    output = model(x, padding_masks)
    print(output.shape)

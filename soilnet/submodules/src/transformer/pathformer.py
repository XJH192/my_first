import math
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import numpy as np
from typing import Any, TypeVar
Tensor = TypeVar('Tensor')
from layers.AMS import AMS
from layers.Layer import WeightGenerator, CustomLinear
from layers.RevIN import RevIN

class PathFormerConfig:
    """配置类，用于设置 PathFormer 模型的参数"""
    def __init__(self, 
                 seq_len=61,          # 输入序列长度
                 pred_len=1,          # 预测长度
                 num_nodes=13,        # 特征维度
                 d_model=64,          # 模型维度
                 d_ff=128,            # 前馈网络维度
                 layer_nums=2,        # 层数
                 num_experts_list=[4, 4],  # 每层专家数量
                 patch_size_list=[[8, 6, 4, 2], [8, 6, 4, 2]],  # 每层的补丁大小
                 k=2,                 # 路由选择的专家数量
                 residual_connection=1,  # 是否使用残差连接
                 revin=False,         # 是否使用 RevIN
                 batch_norm=True,     # 是否使用批归一化
                 gpu=0,               # GPU ID
                 num_classes=1        # 输出类别数
                ):
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_nodes = num_nodes
        self.d_model = d_model
        self.d_ff = d_ff
        self.layer_nums = layer_nums
        self.num_experts_list = num_experts_list
        self.patch_size_list = patch_size_list
        self.k = k
        self.residual_connection = residual_connection
        self.revin = revin
        self.batch_norm = batch_norm
        self.gpu = gpu
        self.num_classes = num_classes

class PathFormerClassiregressor(nn.Module):
    """
    PathFormer 模型，替代 TSTransformerEncoderClassiregressor
    """
    def __init__(
        self,
        feat_dim: int,
        max_len: int,  # 保持这个参数名不变
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        num_classes: int,
        dropout: float = 0.1,
        pos_encoding: str = "fixed",
        activation: str = "gelu",
        norm: str = "BatchNorm",
        freeze: bool = False,
    ):
        super(PathFormerClassiregressor, self).__init__()
        
        # 创建 PathFormer 配置
        self.config = PathFormerConfig(
            seq_len=max_len,  # 使用 max_len 参数
            num_nodes=feat_dim,
            d_model=d_model,
            d_ff=dim_feedforward,
            layer_nums=num_layers,
            num_experts_list=[n_heads] * num_layers,
            patch_size_list=[[8, 6, 4, 2]] * num_layers,
            num_classes=num_classes,
            batch_norm=(norm == "BatchNorm")
        )
        
        # 保存参数供后续使用
        self.max_len = max_len
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        # 初始化 PathFormer 模型
        self.model = PathFormerModel(self.config)
        
        # 输出层，将 PathFormer 的输出映射到类别
        self.output_layer = nn.Linear(max_len * feat_dim, num_classes)
        
        # 激活函数
        if activation == "relu":
            self.act = nn.ReLU()
        elif activation == "gelu":
            self.act = nn.GELU()
        else:
            self.act = nn.ReLU()
            
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X: Tensor, padding_masks: Tensor = None) -> Tensor:
        """
        Args:
            X: (batch_size, seq_length, feat_dim) 输入特征张量
            padding_masks: (batch_size, seq_length) 布尔张量，1表示保留该位置的向量，0表示填充
        Returns:
            output: (batch_size, num_classes)
        """
        # 确保输入形状正确
        if X.shape[1] != self.max_len or X.shape[2] != self.feat_dim:
            X = X.transpose(1, 2)  # 转置为 (batch_size, seq_length, feat_dim)
        
        # 调用 PathFormer 模型
        model_output, _ = self.model(X)  # 输出形状为 (batch_size, pred_len, feat_dim)
        
        # 处理 padding_masks
        if padding_masks is None:
            padding_masks = torch.ones(X.shape[0], X.shape[1], dtype=torch.bool, device=X.device)
        
        # 将模型输出展平
        flattened = model_output.reshape(model_output.shape[0], -1)  # (batch_size, pred_len * feat_dim)
        
        # 应用输出层
        output = self.output_layer(flattened)  # (batch_size, num_classes)
        
        return output

class PathFormerModel(nn.Module):
    """PathFormer 模型的核心实现"""
    def __init__(self, configs):
        super(PathFormerModel, self).__init__()
        self.layer_nums = configs.layer_nums
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu) if torch.cuda.is_available() and torch.cuda.device_count() > 0 else 'cpu')
        self.batch_norm = configs.batch_norm

        for num in range(self.layer_nums):
            self.AMS_lists.append(
                AMS(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1, 
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm))
        
        # 修改投影层，使其适应分类/回归任务
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len * self.num_nodes)
        )

    def forward(self, x):
        balance_loss = 0
        # 归一化
        if self.revin:
            x = self.revin_layer(x, 'norm')
        
        # 将输入扩展为 (batch_size, seq_len, num_nodes, 1)
        out = self.start_fc(x.unsqueeze(-1))
        batch_size = x.shape[0]

        # 通过所有 AMS 层
        for layer in self.AMS_lists:
            out, aux_loss = layer(out)
            balance_loss += aux_loss

        # 重塑输出
        out = out.permute(0, 2, 1, 3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).reshape(batch_size, self.pre_len, self.num_nodes)

        # 反归一化
        if self.revin:
            out = self.revin_layer(out, 'denorm')

        return out, balance_loss

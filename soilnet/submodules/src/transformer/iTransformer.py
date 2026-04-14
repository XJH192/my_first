import torch
import torch.nn as nn
import math
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted

class iTransformerEncoderClassiregressor(nn.Module):
    """
    iTransformer 替代版本的 TSTransformerEncoderClassiregressor
    保持与原始模型相同的接口，但使用 iTransformer 的架构
    """
    def __init__(
        self,
        feat_dim: int,
        max_len: int,
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
        super(iTransformerEncoderClassiregressor, self).__init__()
        
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.feat_dim = feat_dim
        self.num_classes = num_classes
        
        # iTransformer 使用的嵌入层
        self.enc_embedding = DataEmbedding_inverted(
            max_len, d_model, embed_type=pos_encoding, freq='h', dropout=dropout
        )
        
        # 构建编码器层
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 5, attention_dropout=dropout, output_attention=True),
                        d_model, n_heads
                    ),
                    d_model,
                    dim_feedforward,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # 输出层 - 将所有特征连接起来用于分类/回归
        self.output_layer = nn.Linear(d_model * feat_dim, num_classes)
        
        # 是否使用归一化
        self.use_norm = norm != "None"
        
    def forward(self, X: torch.Tensor, padding_masks: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            X: (batch_size, seq_length, feat_dim) 输入张量
            padding_masks: (batch_size, seq_length) 布尔张量，1表示保留，0表示填充
        Returns:
            output: (batch_size, num_classes)
        """
        if padding_masks is None:
            padding_masks = torch.ones(X.shape[0], X.shape[1]).bool().to(X.device)
        
        batch_size = X.shape[0]
        
        # 归一化 (可选)
        if self.use_norm:
            means = X.mean(1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(torch.var(X, dim=1, keepdim=True, unbiased=False) + 1e-5)
            X = X / stdev
        
        # iTransformer 的嵌入 - 注意这里的转置
        # 原始输入: [batch_size, seq_length, feat_dim]
        # 嵌入后: [batch_size, feat_dim, d_model]
        enc_out = self.enc_embedding(X, None)
        
        # 通过编码器
        # 输入: [batch_size, feat_dim, d_model]
        # 输出: [batch_size, feat_dim, d_model]
        enc_out, _ = self.encoder(enc_out)
        
        # 重塑张量用于分类/回归
        # [batch_size, feat_dim, d_model] -> [batch_size, feat_dim * d_model]
        output = enc_out.reshape(batch_size, -1)
        
        # 最终输出层
        output = self.output_layer(output)
        
        return output
class iTransformerEncoder(nn.Module):
    """
    iTransformer 替代版本的 TSTransformerEncoder
    用于序列到序列的任务
    """
    def __init__(
        self,
        feat_dim: int,
        max_len: int,
        d_model: int,
        n_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
        pos_encoding: str = "fixed",
        activation: str = "gelu",
        norm: str = "BatchNorm",
        freeze: bool = False,
    ):
        super(iTransformerEncoder, self).__init__()
        
        self.max_len = max_len
        self.d_model = d_model
        self.n_heads = n_heads
        self.feat_dim = feat_dim
        self.pred_len = max_len  # 预测长度与输入长度相同
        
        # iTransformer 使用的嵌入层
        self.enc_embedding = DataEmbedding_inverted(
            max_len, d_model, embed_type=pos_encoding, freq='h', dropout=dropout
        )
        
        # 构建编码器层
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, 5, attention_dropout=dropout, output_attention=True),
                        d_model, n_heads
                    ),
                    d_model,
                    dim_feedforward,
                    dropout=dropout,
                    activation=activation
                ) for _ in range(num_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model)
        )
        
        # 投影层 - 从 d_model 映射到序列长度
        self.projector = nn.Linear(d_model, max_len, bias=True)
        
        # 是否使用归一化
        self.use_norm = norm != "None"
        
    def forward(self, X: torch.Tensor, padding_masks: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            X: (batch_size, seq_length, feat_dim) 输入张量
            padding_masks: (batch_size, seq_length) 布尔张量，1表示保留，0表示填充
        Returns:
            output: (batch_size, seq_length, feat_dim)
        """
        if padding_masks is None:
            padding_masks = torch.ones(X.shape[0], X.shape[1]).bool().to(X.device)
        
        # 保存原始形状用于后续处理
        batch_size, seq_len, num_vars = X.shape
        
        # 归一化 (可选)
        if self.use_norm:
            means = X.mean(1, keepdim=True).detach()
            X = X - means
            stdev = torch.sqrt(torch.var(X, dim=1, keepdim=True, unbiased=False) + 1e-5)
            X = X / stdev
        
        # iTransformer 的嵌入
        enc_out = self.enc_embedding(X, None)
        
        # 通过编码器
        enc_out, _ = self.encoder(enc_out)
        
        # 投影到序列长度
        # [batch_size, num_vars, d_model] -> [batch_size, num_vars, seq_len]
        dec_out = self.projector(enc_out).permute(0, 2, 1)
        
        # 如果使用了归一化，需要反归一化
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        
        return dec_out

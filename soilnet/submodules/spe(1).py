import torch
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from concurrent.futures import ThreadPoolExecutor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.random_projection import GaussianRandomProjection
from sklearn.kernel_ridge import KernelRidge
import time
# VGG风格的光谱网络
class VGGSpectraNet(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        print(f"Initializing VGGSpectraNet with in_features={in_features}, hidden_size={hidden_size}")
        
        self.features = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            
            nn.Linear(128, hidden_size),
            nn.ReLU()
        )
    
    def forward(self, x):
        # Ensure correct dimensions
        if len(x.shape) == 3:
            x = x.squeeze(1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        return self.features(x)  # 只返回一个输出




# ResNet风格的光谱网络
class ResSpectraBlock(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.BatchNorm1d(out_features)
        )
        self.shortcut = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features)
        ) if in_features != out_features else nn.Identity()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.block(x)
        return self.relu(out + identity)

class ResSpectraNet(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            ResSpectraBlock(512, 256),
            ResSpectraBlock(256, hidden_size)
        )
    
    def forward(self, x):
        return self.features(x)

# DenseNet风格的光谱网络
class DenseSpectraNet(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        self.dense1 = nn.Linear(in_features, 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256 + 512, hidden_size)
        
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        d1 = self.relu(self.bn1(self.dense1(x)))
        d1 = self.dropout(d1)
        
        d2 = self.relu(self.bn2(self.dense2(d1)))
        d2 = self.dropout(d2)
        
        d3_input = torch.cat([d1, d2], dim=1)
        d3 = self.relu(self.dense3(d3_input))
        
        return d3

class LSTM(nn.Module):
    def __init__(self, in_features, hidden_size=16, num_layers=2):
        super(LSTM, self).__init__()
        print(f"Initializing CNNLSTMNet with in_features={in_features}, hidden_size={hidden_size}")
        self.input_size = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN参数设置
        self.conv_size = 64
        
        # CNN层处理原始数据
        self.conv = nn.Conv1d(1, self.conv_size, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn = nn.BatchNorm1d(self.conv_size)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # CNN特征投影层
        self.cnn_projection = nn.Linear(self.conv_size, hidden_size)
        
        # LSTM处理序列数据
        self.lstm = nn.LSTM(
            input_size=hidden_size,  # 输入是CNN特征
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.01,
            bidirectional=True  # 使用双向LSTM
        )
        
        # 全连接层
        self.fc_1 = nn.Linear(hidden_size * 2, hidden_size * 2)  # 因为双向LSTM，维度翻倍
        self.fc_2 = nn.Linear(hidden_size * 2, hidden_size)
        
        # 其他层
        self.dropout = nn.Dropout(p=0.02)

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        
        # 重塑输入数据为CNN期望的格式: [batch_size, 1, sequence_length]
        x = x.view(batch_size, 1, -1)
        
        # CNN处理
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # 重塑为序列数据，准备输入LSTM
        # 从[batch_size, conv_size, reduced_length]转换为[batch_size, reduced_length, conv_size]
        x = x.transpose(1, 2)
        reduced_length = x.size(1)
        
        # 将CNN特征投影到LSTM所需的维度
        x = self.cnn_projection(x)  # [batch_size, reduced_length, hidden_size]
        
        # 初始化LSTM隐藏状态和单元状态
        hidden_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)  # *2因为双向
        cell_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        
        # LSTM处理序列
        lstm_out, (hidden_n, _) = self.lstm(x, (hidden_0, cell_0))  # [batch_size, reduced_length, hidden_size*2]
        
        # 使用最后一个时间步的输出
        last_output = lstm_out[:, -1, :]  # [batch_size, hidden_size*2]
        
        # 全连接层处理
        x = self.dropout(self.relu(self.fc_1(last_output)))
        x = self.fc_2(x)  # 输出维度为 [batch_size, hidden_size]
        
        return x

class PLSRNet(nn.Module):
    def __init__(self, in_features, hidden_size, n_components=None):
        """
        PLS回归与神经网络结合的光谱分析模型
        
        参数:
            in_features: 输入光谱特征维度
            hidden_size: 隐藏层维度
            n_components: PLS成分数量，默认为None时自动设置
        """
        super().__init__()
        
        # 确定PLS成分数量
        if n_components is None:
            self.n_components = min(in_features // 2, hidden_size)
        else:
            self.n_components = n_components
            
        # 初始化PLS模型
        self.pls_model = PLSRegression(n_components=self.n_components)
        
        # 微调层：将PLS特征映射到最终输出
        self.fine_tune = nn.Sequential(
            nn.Linear(self.n_components, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # 注册缓冲区存储PLS参数和训练状态
        self.register_buffer('x_weights', torch.zeros(in_features, self.n_components))
        self.register_buffer('x_loadings', torch.zeros(in_features, self.n_components))
        self.register_buffer('y_weights', torch.zeros(1, self.n_components))
        self.register_buffer('x_scores', torch.zeros(1, self.n_components))
        self.register_buffer('y_loadings', torch.zeros(1, self.n_components))
        self.register_buffer('x_rotations', torch.zeros(in_features, self.n_components))
        self.register_buffer('y_rotations', torch.zeros(1, self.n_components))
        self.register_buffer('coef_', torch.zeros(in_features, 1))
        self.register_buffer('intercept_', torch.zeros(1))
        self.register_buffer('pls_trained', torch.tensor([False], dtype=torch.bool))
        
    def fit_pls(self, X_tensor, Y_tensor):
        """
        使用输入的光谱数据和目标值训练PLS模型
        
        参数:
            X_tensor: 光谱数据张量 [batch_size, n_features] 或 [batch_size, seq_len, n_features]
            Y_tensor: 目标值张量
        """
        # 记录输入数据的设备
        device = X_tensor.device
        
        # 将张量转换为numpy数组
        X_np = X_tensor.cpu().detach().numpy()
        Y_np = Y_tensor.cpu().detach().numpy()
        
        # 处理三维数据（如果输入是序列）
        if X_np.ndim > 2:
            original_shape = X_np.shape
            X_np = X_np.reshape(-1, original_shape[-1])
            
            # 调整目标值形状以匹配
            if Y_np.ndim > 1:
                Y_np = Y_np.reshape(-1, Y_np.shape[-1]) if Y_np.shape[-1] > 1 else Y_np.reshape(-1)
        
        # 训练PLS模型
        self.pls_model.fit(X_np, Y_np)
        
        # 保存PLS参数到缓冲区，并移到与输入数据相同的设备
        self.x_weights = torch.tensor(self.pls_model.x_weights_, dtype=torch.float32, device=device)
        self.x_loadings = torch.tensor(self.pls_model.x_loadings_, dtype=torch.float32, device=device)
        self.y_weights = torch.tensor(self.pls_model.y_weights_, dtype=torch.float32, device=device)
        self.x_scores = torch.tensor(self.pls_model.x_scores_, dtype=torch.float32, device=device)
        self.y_loadings = torch.tensor(self.pls_model.y_loadings_, dtype=torch.float32, device=device)
        self.x_rotations = torch.tensor(self.pls_model.x_rotations_, dtype=torch.float32, device=device)
        self.y_rotations = torch.tensor(self.pls_model.y_rotations_, dtype=torch.float32, device=device)
        self.coef_ = torch.tensor(self.pls_model.coef_, dtype=torch.float32, device=device)
        self.intercept_ = torch.tensor([self.pls_model.intercept_], dtype=torch.float32, device=device)
        
        # 标记PLS已训练
        self.pls_trained.fill_(True)
        print(f"PLS模型训练完成，使用{self.n_components}个成分")
        
    def forward(self, x):
        """
        前向传播：先进行PLS变换，再通过微调层
        
        参数:
            x: 输入光谱数据 [batch_size, n_features] 或 [batch_size, seq_len, n_features]
        """
        # 确保PLS模型已训练
        if not self.pls_trained.item():
            raise RuntimeError("PLS模型尚未训练，请先调用fit_pls方法")
        
        # 确保所有PLS参数与输入数据在同一设备上
        if x.device != self.x_rotations.device:
            self.x_weights = self.x_weights.to(x.device)
            self.x_loadings = self.x_loadings.to(x.device)
            self.y_weights = self.y_weights.to(x.device)
            self.x_scores = self.x_scores.to(x.device)
            self.y_loadings = self.y_loadings.to(x.device)
            self.x_rotations = self.x_rotations.to(x.device)
            self.y_rotations = self.y_rotations.to(x.device)
            self.coef_ = self.coef_.to(x.device)
            self.intercept_ = self.intercept_.to(x.device)
        
        # 处理不同维度的输入
        if x.dim() == 2:
            # 二维输入: [batch_size, n_features]
            batch_size, n_features = x.shape
            
            # 应用PLS变换
            x_pls = torch.matmul(x, self.x_rotations)
            
        elif x.dim() == 3:
            # 三维输入: [batch_size, seq_len, n_features]
            batch_size, seq_len, n_features = x.shape
            
            # 调整形状以应用PLS变换
            x_reshaped = x.reshape(batch_size * seq_len, n_features)
            x_pls_reshaped = torch.matmul(x_reshaped, self.x_rotations)
            
            # 恢复序列维度
            x_pls = x_pls_reshaped.reshape(batch_size, seq_len, self.n_components)
            
            # 对于序列数据，可以选择取平均值或使用其他聚合方法
            x_pls = x_pls.mean(dim=1)  # 取序列平均值
            
        else:
            raise ValueError(f"输入维度 {x.dim()} 不支持，应为2或3")
        
        # 通过微调层
        output = self.fine_tune(x_pls)
        
        return output



class SVM(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        print(f"Initializing SVM with in_features={in_features}, hidden_size={hidden_size}")
        
        self.hidden_size = hidden_size
        self.svr = MultiOutputRegressor(SVR(kernel='rbf', C=1.0, gamma='scale'))
        self.is_fitted = False
        
        # 用于将SVM输出映射到hidden_size
        self.mapper = nn.Linear(1, hidden_size)  # 从SVR的单一输出映射到hidden_size
        
        # 添加一个标志来跟踪SVM是否有效
        self.svm_effective = False
        # 存储训练和预测的误差，用于评估SVM效果
        self.training_error = None
        
    def fit(self, X, y=None):
        """训练SVR模型，使用实际标签而不是随机数据"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        if y is not None:
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
            # 使用实际标签进行拟合
            target = y
        else:
            # 如果没有提供标签，使用X的均值作为简单目标
            # 这比随机值更有意义，可以捕捉数据的某些特征
            target = np.mean(X, axis=1, keepdims=True)
        
        # 保存原始目标用于评估
        self.original_target = target
        
        # 训练SVR
        self.svr.fit(X, target)
        self.is_fitted = True
        
        # 计算训练误差
        pred = self.svr.predict(X)
        self.training_error = np.mean((pred - target) ** 2)
        print(f"SVM training error: {self.training_error:.6f}")
        
        # 如果训练误差小于阈值，认为SVM有效
        self.svm_effective = self.training_error < 0.5  # 可以调整阈值
        print(f"SVM is effective: {self.svm_effective}")
        
        return self
    
    def forward(self, x):
        # 确保维度正确
        if len(x.shape) == 3:
            x = x.squeeze(1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        if not self.is_fitted:
            # 如果SVR未训练，则进行初始拟合
            self.fit(x)
        
        # 转换为numpy并预测
        x_np = x.detach().cpu().numpy()
        
        # 使用SVR进行预测
        pred = self.svr.predict(x_np)
        pred_tensor = torch.FloatTensor(pred).to(x.device)
        
        # 映射到所需的hidden_size
        output = self.mapper(pred_tensor)
        
        # 记录SVM的有效性但不改变返回值类型
        # 只打印信息，不改变返回值结构
        if hasattr(self, 'svm_effective') and hasattr(self, 'training_error'):
            # 可以在这里添加日志记录，但不改变返回值
            if not hasattr(self, '_logged_once'):
                print(f"SVM effectiveness in forward pass: {self.svm_effective}, error: {self.training_error:.6f}")
                self._logged_once = True
        
        # 只返回张量，不返回元组
        return output
    
    def is_effective(self):
        """返回SVM是否有效的方法"""
        if not hasattr(self, 'svm_effective'):
            return False, None
        return self.svm_effective, self.training_error if hasattr(self, 'training_error') else None


class CNNNet(nn.Module):
    def __init__(self, in_features, hidden_size=128, num_layers=2):
        super(CNNNet, self).__init__()
        print(f"Initializing CNNNet with in_features={in_features}, hidden_size={hidden_size}")
        
        # 保留hidden_size作为最终输出维度
        self.hidden_size = hidden_size
        
        # CNN参数设置
        self.conv1_size = 64
        self.conv2_size = 128
        self.conv3_size = 256
        self.conv4_size = 384
        self.conv5_size = 512

        # CNN层 - 第一层输入通道为1，因为我们处理的是一维信号
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, self.conv1_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv1_size, self.conv2_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv2_size, self.conv3_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv3_size, self.conv4_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv4_size, self.conv5_size, kernel_size=3, padding=1, stride=2, bias=False)
        ])

        # BatchNorm层
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(self.conv1_size),
            nn.BatchNorm1d(self.conv2_size),
            nn.BatchNorm1d(self.conv3_size),
            nn.BatchNorm1d(self.conv4_size),
            nn.BatchNorm1d(self.conv5_size)
        ])

        # 池化层
        self.avgpool = nn.AvgPool1d(kernel_size=2)
        self.adp_avgpool = nn.AdaptiveAvgPool1d(1)

        # 全连接层，保持最终输出维度为hidden_size
        self.fc_1 = nn.Linear(self.conv5_size, self.conv2_size)
        self.fc_2 = nn.Linear(self.conv2_size, hidden_size)

        # 激活函数和dropout
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout1d(p=0.02)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 重塑输入数据为CNN期望的格式: [batch_size, channels, length]
        # 假设输入是[batch_size, sequence_length]
        x = x.view(batch_size, 1, -1)
        
        # CNN处理
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
        
        # 最终处理
        x = self.adp_avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.fc_1(x)))
        x = self.fc_2(x)  # 输出维度为 [batch_size, hidden_size]
        
        return x


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention_weights = None
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1))
        self.attention_weights = weights
        return outputs

class LSTMCNNAttentionNet(nn.Module):
    def __init__(self, in_features, hidden_size=16, num_layers=2):
        super(LSTMCNNAttentionNet, self).__init__()
        print(f"Initializing CNNLSTMAttentionNet with in_features={in_features}, hidden_size={hidden_size}")
        self.input_size = 1
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # CNN参数设置
        self.conv1_size = 64
        self.conv2_size = 128
        self.conv3_size = 256
        self.conv4_size = 384
        self.conv5_size = 512

        # 第一阶段：CNN处理原始数据
        self.cnn_first_stage = nn.ModuleList([
            nn.Conv1d(1, self.conv1_size, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm1d(self.conv1_size),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        ])
        
        # CNN特征提取后的投影层，用于将CNN特征映射到LSTM输入维度
        self.cnn_projection = nn.Linear(self.conv1_size, hidden_size)
        
        # 第二阶段：LSTM处理全局序列信息
        self.lstm = nn.LSTM(
            input_size=hidden_size,  # 输入是CNN特征
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.01,
            bidirectional=True  # 使用双向LSTM更好地捕获全局信息
        )
        
        # 第三阶段：Attention关注局部特征
        self.attention = SelfAttention(hidden_size * 2)  # 因为使用双向LSTM，维度翻倍
        
        # 深度特征提取：更多CNN层
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(hidden_size * 2, self.conv2_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv2_size, self.conv3_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv3_size, self.conv4_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv4_size, self.conv5_size, kernel_size=3, padding=1, stride=2, bias=False)
        ])

        # BatchNorm层
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(self.conv2_size),
            nn.BatchNorm1d(self.conv3_size),
            nn.BatchNorm1d(self.conv4_size),
            nn.BatchNorm1d(self.conv5_size)
        ])

        # 池化层
        self.avgpool = nn.AvgPool1d(kernel_size=2)
        self.adp_avgpool = nn.AdaptiveAvgPool1d(1)

        # 全连接层
        self.fc_1 = nn.Linear(self.conv5_size, self.conv2_size)
        self.fc_2 = nn.Linear(self.conv2_size, hidden_size)

        # 其他层
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.02)

    def forward(self, x):
        device = x.device
        batch_size = x.shape[0]
        
        # 重塑输入数据为CNN期望的格式: [batch_size, 1, sequence_length]
        x = x.view(batch_size, 1, -1)
        sequence_length = x.size(2)
        
        # 第一阶段：CNN处理原始数据
        # 应用第一个CNN层、BatchNorm、ReLU和池化
        for layer in self.cnn_first_stage:
            x = layer(x)
        
        # 重塑为序列数据，准备输入LSTM
        # 从[batch_size, conv1_size, reduced_length]转换为[batch_size, reduced_length, conv1_size]
        x = x.transpose(1, 2)
        reduced_length = x.size(1)
        
        # 将CNN特征投影到LSTM所需的维度
        x = self.cnn_projection(x)  # [batch_size, reduced_length, hidden_size]
        
        # 第二阶段：LSTM处理全局序列信息
        # 初始化LSTM隐藏状态和单元状态
        hidden_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)  # *2因为双向
        cell_0 = torch.zeros(self.num_layers * 2, batch_size, self.hidden_size).to(device)
        
        lstm_out, (_, _) = self.lstm(x, (hidden_0, cell_0))  # [batch_size, reduced_length, hidden_size*2]
        
        # 第三阶段：Attention关注局部特征
        att_out = self.attention(lstm_out)  # [batch_size, reduced_length, hidden_size*2]
        
        # 准备进入深度CNN处理
        # 从[batch_size, reduced_length, hidden_size*2]转换为[batch_size, hidden_size*2, reduced_length]
        x = att_out.transpose(1, 2)
        
        # 深度特征提取：更多CNN层
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
        
        # 最终处理
        x = self.adp_avgpool(x)
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.fc_1(x)))
        x = self.fc_2(x)  # 输出维度为 [batch_size, hidden_size]
        
        return x

class LSSVM(nn.Module):
    def __init__(self, in_features, hidden_size):
        super().__init__()
        print(f"Initializing LSSVM with in_features={in_features}, hidden_size={hidden_size}")
        
        self.hidden_size = hidden_size
        # 将SVR替换为LSSVM (使用sklearn的LSSVRegressor)
        # 由于sklearn没有直接的LSSVM实现，我们使用KernelRidge作为替代，它是LSSVM的一个变种
        self.lssvr = MultiOutputRegressor(KernelRidge(kernel='rbf', alpha=1.0, gamma=None))
        self.is_fitted = False
        
        # 用于将LSSVM输出映射到hidden_size
        self.mapper = nn.Linear(1, hidden_size)  # 从LSSVR的单一输出映射到hidden_size
        
        # 添加一个标志来跟踪LSSVM是否有效
        self.lssvm_effective = False
        # 存储训练和预测的误差，用于评估LSSVM效果
        self.training_error = None
        
    def fit(self, X, y=None):
        """训练LSSVR模型，使用实际标签而不是随机数据"""
        if isinstance(X, torch.Tensor):
            X = X.detach().cpu().numpy()
        
        if y is not None:
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
            # 使用实际标签进行拟合
            target = y
        else:
            # 如果没有提供标签，使用X的均值作为简单目标
            # 这比随机值更有意义，可以捕捉数据的某些特征
            target = np.mean(X, axis=1, keepdims=True)
        
        # 保存原始目标用于评估
        self.original_target = target
        
        # 训练LSSVR
        self.lssvr.fit(X, target)
        self.is_fitted = True
        
        # 计算训练误差
        pred = self.lssvr.predict(X)
        self.training_error = np.mean((pred - target) ** 2)
        print(f"LSSVM training error: {self.training_error:.6f}")
        
        # 如果训练误差小于阈值，认为LSSVM有效
        self.lssvm_effective = self.training_error < 0.5  # 可以调整阈值
        print(f"LSSVM is effective: {self.lssvm_effective}")
        
        return self
    
    def forward(self, x):
        # 确保维度正确
        if len(x.shape) == 3:
            x = x.squeeze(1)
        elif len(x.shape) == 1:
            x = x.unsqueeze(0)
        
        batch_size = x.shape[0]
        
        if not self.is_fitted:
            # 如果LSSVR未训练，则进行初始拟合
            self.fit(x)
        
        # 转换为numpy并预测
        x_np = x.detach().cpu().numpy()
        
        # 使用LSSVR进行预测
        pred = self.lssvr.predict(x_np)
        pred_tensor = torch.FloatTensor(pred).to(x.device)
        
        # 映射到所需的hidden_size
        output = self.mapper(pred_tensor)
        
        # 记录LSSVM的有效性但不改变返回值类型
        # 只打印信息，不改变返回值结构
        if hasattr(self, 'lssvm_effective') and hasattr(self, 'training_error'):
            # 可以在这里添加日志记录，但不改变返回值
            if not hasattr(self, '_logged_once'):
                print(f"LSSVM effectiveness in forward pass: {self.lssvm_effective}, error: {self.training_error:.6f}")
                self._logged_once = True
        
        # 只返回张量，不返回元组
        return output
    
    def is_effective(self):
        """返回LSSVM是否有效的方法"""
        if not hasattr(self, 'lssvm_effective'):
            return False, None
        return self.lssvm_effective, self.training_error if hasattr(self, 'training_error') else None

class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.attention_weights = None
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, encoder_outputs):
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        outputs = (encoder_outputs * weights.unsqueeze(-1))
        self.attention_weights = weights
        return outputs

class CNNNetWithAttention(nn.Module):
    def __init__(self, in_features, hidden_size=128, num_layers=2):
        super(CNNNetWithAttention, self).__init__()
        print(f"Initializing CNNNetWithAttention with in_features={in_features}, hidden_size={hidden_size}")
        
        # 保留hidden_size作为最终输出维度
        self.hidden_size = hidden_size
        
        # CNN参数设置
        self.conv1_size = 64
        self.conv2_size = 128
        self.conv3_size = 256
        self.conv4_size = 384
        self.conv5_size = 512

        # CNN层 - 第一层输入通道为1，因为我们处理的是一维信号
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(1, self.conv1_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv1_size, self.conv2_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv2_size, self.conv3_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv3_size, self.conv4_size, kernel_size=3, padding=1, stride=2, bias=False),
            nn.Conv1d(self.conv4_size, self.conv5_size, kernel_size=3, padding=1, stride=2, bias=False)
        ])

        # BatchNorm层
        self.bn_layers = nn.ModuleList([
            nn.BatchNorm1d(self.conv1_size),
            nn.BatchNorm1d(self.conv2_size),
            nn.BatchNorm1d(self.conv3_size),
            nn.BatchNorm1d(self.conv4_size),
            nn.BatchNorm1d(self.conv5_size)
        ])

        # 池化层
        self.avgpool = nn.AvgPool1d(kernel_size=2)
        
        # 注意：我们不再使用自适应平均池化，而是保留特征序列用于注意力机制
        # self.adp_avgpool = nn.AdaptiveAvgPool1d(1)
        
        # 添加自注意力模块
        self.attention = SelfAttention(self.conv5_size)
        
        # 全连接层，保持最终输出维度为hidden_size
        self.fc_1 = nn.Linear(self.conv5_size, self.conv2_size)
        self.fc_2 = nn.Linear(self.conv2_size, hidden_size)

        # 激活函数和dropout
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout1d(p=0.02)

    def forward(self, x):
        batch_size = x.shape[0]
        
        # 重塑输入数据为CNN期望的格式: [batch_size, channels, length]
        # 假设输入是[batch_size, sequence_length]
        x = x.view(batch_size, 1, -1)
        
        # CNN处理
        for conv, bn in zip(self.conv_layers, self.bn_layers):
            x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
        
        # 转换为注意力机制需要的格式 [batch_size, seq_len, features]
        x = x.transpose(1, 2)  # 从 [batch_size, features, seq_len] 到 [batch_size, seq_len, features]
        
        # 应用自注意力机制
        x = self.attention(x)
        
        # 聚合特征 - 对注意力加权后的特征进行求和
        x = torch.sum(x, dim=1)  # [batch_size, features]
        
        # 全连接层处理
        x = self.dropout(self.relu(self.fc_1(x)))
        x = self.fc_2(x)  # 输出维度为 [batch_size, hidden_size]
        
        return x

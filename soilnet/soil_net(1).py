import torch
import torch.nn as nn
from submodules.cnn_feature_extractor import CNNFlattener64, CNNFlattener128,\
                                                ResNet101, ResNet101GLAM,\
                                                ResNet50,\
                                                    VGG16, VGG16GLAM
from submodules.vit import VisionTransformer as ViT
from submodules.regressor import Regressor, MultiHeadRegressor
from submodules import rnn
from typing import Tuple
from submodules.src.transformer.transformer import TSTransformerEncoderClassiregressor
from submodules.src.transformer.iTransformer import iTransformerEncoderClassiregressor
from submodules.spe import VGGSpectraNet,ResSpectraNet,DenseSpectraNet,LSTMCNNAttentionNet,LSTM,PLSRNet,SVM,MambaCNNAttentionNet,RFSpectraNet,CNNNet,LSSVM,CNNNetWithAttention
import torch.nn.functional as F
class SoilNet(nn.Module):
    def __init__(self, use_glam = False , cnn_arch = "resnet101", reg_version = 1,
                 cnn_in_channels = 14 ,regresor_input_from_cnn = 1024, hidden_size=128, img_size = 64):
        super().__init__()
        if use_glam:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                raise ValueError("ViT is not supported when GLAM is enabled. Please choose from 'resnet' or 'vgg16' or disable GLAM.")
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16' or 'ViT'.")

        else:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "resnet50":
                self.cnn = ResNet50(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                self.cnn = ViT(img_size=img_size, patch_size=8, in_chans=cnn_in_channels, n_classes=regresor_input_from_cnn, p=0.1, attn_p=0.1)
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16' or 'ViT'.")
            

        
        self.reg = MultiHeadRegressor(regresor_input_from_cnn, hidden_size= hidden_size, version=reg_version)
    def forward(self, raster_stack):
        """
        Forward pass of the Resnet module.
        
        Args:
            raster_stack (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
            auxillary_data (torch.Tensor): Auxiliary input tensor of shape (batch_size, aux_size).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        flat_raster = self.cnn(raster_stack)
        output = self.reg(flat_raster)
        return output
        
class SoilNetLSTM(nn.Module):
    def __init__(self, use_glam = False  , cnn_arch = "resnet101", reg_version = 1,
                 cnn_in_channels = 14 ,regresor_input_from_cnn = 1024, 
                 lstm_n_features = 10,lstm_n_layers =2, lstm_out = 128, hidden_size=128, rnn_arch = "LSTM", seq_len = 61, img_size = 64):
        
        super().__init__()
        
        if use_glam:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                raise ValueError("ViT is not supported when GLAM is enabled. Please choose from 'resnet' or 'vgg16' or disable GLAM.")
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16'.")

        else:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "resnet50":
                self.cnn = ResNet50(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                self.cnn = ViT(img_size=img_size, patch_size=8, in_chans=cnn_in_channels, n_classes=regresor_input_from_cnn, p=0.1, attn_p=0.1)
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16'.")
            

        if rnn_arch == "LSTM":
            self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "GRU":
            self.lstm = rnn.GRU(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "RNN":
            self.lstm = rnn.RNN(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "Transformer":
            self.lstm = TSTransformerEncoderClassiregressor(
                        feat_dim=lstm_n_features,
                        max_len=61,
                        d_model=512,
                        n_heads=8,
                        num_layers=6,
                        dim_feedforward=2048, 
                        num_classes=lstm_out,
                        dropout=0.1,
                        pos_encoding="fixed",
                        activation="gelu",
                        norm="BatchNorm",
                        freeze=False,
                        )
        elif rnn_arch == "itransformer":
            self.lstm = iTransformerEncoderClassiregressor(
                        feat_dim=lstm_n_features,
                        max_len=61,
                        d_model=512,
                        n_heads=8,
                        num_layers=6,
                        dim_feedforward=2048, 
                        num_classes=lstm_out,
                        dropout=0.1,
                        pos_encoding="fixed",
                        activation="gelu",
                        norm="BatchNorm",
                        freeze=False,
                        )
        else:
            raise ValueError("Invalid RNN Architecture. Please choose from 'LSTM', 'GRU' or 'RNN'.")
        
        self.reg = MultiHeadRegressor(regresor_input_from_cnn, lstm_out, hidden_size= hidden_size, version=reg_version)
        
    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Inputs
        ------
        input_raster_ts : A tupple containing the following two tensors:
            * raster_stack (torch.Tensor): A 4D tensor of shape `(batch_size, channels, height, width)` representing a stack of raster images.
            * ts_features (torch.Tensor): A 3D tensor of shape `(batch_size, seq_length, , n_features)` representing a sequence of time-series features. | `seq_length` is the number of time steps in the sequence. e.g. months in our climate data
            
        Outputs
        -------
            - output (torch.Tensor): A tensor of shape `(batch_size, 1)` representing the predicted output of regression.
        """
        raster_stack, ts_features = input_raster_ts
        flat_raster = self.cnn(raster_stack)
        lstm_output = self.lstm(ts_features)
        output = self.reg(flat_raster, lstm_output)
        return output
    
class SoilNetJustLSTM(SoilNetLSTM):
    """
    This class inherits from SoilNetLSTM but disables the CNN pathway to use only the climate data.
    """
    def __init__(self, use_glam=False, cnn_arch="resnet101", reg_version=1,
                 cnn_in_channels=14, regresor_input_from_cnn=1024, 
                 lstm_n_features=10, lstm_n_layers=2, lstm_out=128, hidden_size=128,
                 rnn_arch="LSTM", seq_len=61, img_size=64):
        
        super().__init__(use_glam=use_glam, cnn_arch=cnn_arch, reg_version=reg_version,
                         cnn_in_channels=cnn_in_channels, regresor_input_from_cnn=regresor_input_from_cnn, 
                         lstm_n_features=lstm_n_features, lstm_n_layers=lstm_n_layers, lstm_out=lstm_out, hidden_size=hidden_size,
                         rnn_arch=rnn_arch, seq_len=seq_len, img_size=img_size)
    

        self.cnn = None
        self.reg = MultiHeadRegressor(lstm_out, hidden_size= hidden_size, version=reg_version)
            
    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Redefine the forward method if the behavior needs to change for JustLSTM.
        """
        # Handle case where image data isn't used
        _, ts_features = input_raster_ts
        lstm_output = self.lstm(ts_features)
        output = self.reg(lstm_output)  # Assuming reg can handle this case

        return output
class SoilNetCORR(nn.Module):
    def __init__(self, use_glam=False, cnn_arch="resnet101", reg_version=1,
                 cnn_in_channels=14, regresor_input_from_cnn=1024,
                 lstm_n_features=10, lstm_n_layers=2, lstm_out=128,
                 spectral_in_features=2000, spectral_hidden=512, # 新增光谱数据参数
                 hidden_size=128, rnn_arch="LSTM", seq_len=61, img_size=64,spectral_net_type='vgg'):
        
        super().__init__()
        
        # CNN 部分保持不变
        if use_glam:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                raise ValueError("ViT is not supported when GLAM is enabled.")
            else:
                raise ValueError("Invalid CNN Architecture.")
        else:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "resnet50":
                self.cnn = ResNet50(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                self.cnn = ViT(img_size=img_size, patch_size=8, in_chans=cnn_in_channels, 
                             n_classes=regresor_input_from_cnn, p=0.1, attn_p=0.1)
                print("1111")
            else:
                raise ValueError("Invalid CNN Architecture.")

        # LSTM/Transformer 部分保持不变
        if rnn_arch == "LSTM":
            self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "GRU":
            self.lstm = rnn.GRU(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "RNN":
            self.lstm = rnn.RNN(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "Transformer":
            self.lstm = TSTransformerEncoderClassiregressor(
                        feat_dim=lstm_n_features,
                        max_len=61,
                        d_model=512,
                        n_heads=8,
                        num_layers=6,
                        dim_feedforward=2048, 
                        num_classes=lstm_out,
                        dropout=0.1,
                        pos_encoding="fixed",
                        activation="gelu",
                        norm="BatchNorm",
                        freeze=False,
                        )
        elif rnn_arch == 'itransformer':
            self.lstm = iTransformerEncoderClassiregressor(
                        feat_dim=lstm_n_features,
                        max_len=61,
                        d_model=512,
                        n_heads=8,
                        num_layers=6,
                        dim_feedforward=2048, 
                        num_classes=lstm_out,
                        dropout=0.1,
                        pos_encoding="fixed",
                        activation="gelu",
                        norm="BatchNorm",
                        freeze=False,
                        )
        else:
            raise ValueError("Invalid RNN Architecture.")

        # 新增：光谱数据处理网络
    # 选择光谱网络架构
        if spectral_net_type == "vgg":
            self.spectral_net = VGGSpectraNet(spectral_in_features, hidden_size)
        elif spectral_net_type == "resnet":
            self.spectral_net = ResSpectraNet(spectral_in_features, hidden_size)
        elif spectral_net_type == "densenet":
            self.spectral_net = DenseSpectraNet(spectral_in_features, hidden_size)
        elif spectral_net_type == 'lstm':
            self.spectral_net = LSTM(spectral_in_features, hidden_size)
        elif spectral_net_type == "lstmcnn":
            self.spectral_net = LSTMCNNAttentionNet(spectral_in_features, hidden_size)
        elif spectral_net_type == "PLSR":
            self.spectral_net = PLSRNet(spectral_in_features,hidden_size)
        elif spectral_net_type == "svr":
            self.spectral_net = SVM(spectral_in_features,hidden_size)
        elif spectral_net_type == "mambacnn":
            self.spectral_net = MambaCNNAttentionNet(spectral_in_features,hidden_size)
        elif spectral_net_type == "rf":
            self.spectral_net = RFSpectraNet(spectral_in_features,hidden_size)
        elif spectral_net_type == "cnn":
            self.spectral_net = CNNNet(spectral_in_features,hidden_size)
        elif spectral_net_type == "lssvm":
            self.spectral_net = LSSVM(spectral_in_features,hidden_size)
        elif spectral_net_type == "cnnatten":
            self.spectral_net = CNNNetWithAttention(spectral_in_features,hidden_size)
        else:
            # 默认简单MLP
            self.spectral_net = nn.Sequential(
                nn.Linear(spectral_in_features, spectral_hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(spectral_hidden, hidden_size),
                nn.ReLU()
            )

        # 修改回归器以处理三种特征
        self.reg = MultiHeadRegressor(
            regresor_input_from_cnn,  # CNN特征
            lstm_out,                 # LSTM特征
            hidden_size,             # 光谱特征
            hidden_size=hidden_size,
            version=reg_version
        )
    def forward(self, input_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        处理三种输入数据：遥感图像、时序气候数据和光谱数据
        
        参数:
        ------
        input_data: 包含三个张量的元组:
            * raster_stack (torch.Tensor): 形状为 (batch_size, channels, height, width) 的遥感图像
            * ts_features (torch.Tensor): 形状为 (batch_size, seq_length, n_features) 的时序特征
            * spectral_data (torch.Tensor): 形状为 (batch_size, spectral_features) 的光谱数据
            
        返回:
        -------
            output (torch.Tensor): 形状为 (batch_size, 1) 的预测结果
        """
        raster_stack, ts_features, spectral_data = input_data
        
        # 处理遥感图像
        flat_raster = self.cnn(raster_stack)
        
        # 处理时序数据
        lstm_output = self.lstm(ts_features)
        
        # 处理光谱数据
        spectral_features = self.spectral_net(spectral_data)
        
        # 融合三种特征并进行预测
        output = self.reg(flat_raster, lstm_output, spectral_features)
        
        return output
# class SoilNetCORR(nn.Module):
#     def __init__(self, use_glam=False, cnn_arch="resnet101", reg_version=1,
#                  cnn_in_channels=14, regresor_input_from_cnn=1024,
#                  lstm_n_features=10, lstm_n_layers=2, lstm_out=128,
#                  spectral_in_features=2000, spectral_hidden=512, # 光谱数据参数
#                  hidden_size=128, rnn_arch="LSTM", seq_len=61, img_size=64, spectral_net_type='vgg',
#                  # 新增权重参数
#                  spectral_weight=0.6, image_weight=0.2, climate_weight=0.2):
        
#         super().__init__()
        
#         # 存储权重参数
#         self.spectral_weight = spectral_weight
#         self.image_weight = image_weight
#         self.climate_weight = climate_weight
        
#         # CNN 部分保持不变
#         if use_glam:
#             if cnn_arch == "resnet101":
#                 self.cnn = ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
#             elif cnn_arch == "vgg16":
#                 self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
#             elif cnn_arch == "ViT":
#                 raise ValueError("ViT is not supported when GLAM is enabled.")
#             else:
#                 raise ValueError("Invalid CNN Architecture.")
#         else:
#             if cnn_arch == "resnet101":
#                 self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
#             elif cnn_arch == "resnet50":
#                 self.cnn = ResNet50(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
#             elif cnn_arch == "vgg16":
#                 self.cnn = VGG16GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
#             elif cnn_arch == "ViT":
#                 self.cnn = ViT(img_size=img_size, patch_size=8, in_chans=cnn_in_channels, 
#                              n_classes=regresor_input_from_cnn, p=0.1, attn_p=0.1)
#             elif cnn_arch == "MiV":
#                 self.cnn = MiV(img_size=img_size,patch_size=8,in_chans=cnn_in_channels,n_classes=regresor_input_from_cnn)
#                 print("1111")
#             else:
#                 raise ValueError("Invalid CNN Architecture.")

#         # LSTM/Transformer 部分保持不变
#         if rnn_arch == "LSTM":
#             self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
#         elif rnn_arch == "GRU":
#             self.lstm = rnn.GRU(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
#         elif rnn_arch == "RNN":
#             self.lstm = rnn.RNN(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
#         elif rnn_arch == "Transformer":
#             self.lstm = TSTransformerEncoderClassiregressor(
#                         feat_dim=lstm_n_features,
#                         max_len=61,
#                         d_model=512,
#                         n_heads=8,
#                         num_layers=6,
#                         dim_feedforward=2048, 
#                         num_classes=lstm_out,
#                         dropout=0.1,
#                         pos_encoding="fixed",
#                         activation="gelu",
#                         norm="BatchNorm",
#                         freeze=False,
#                         )
#         elif rnn_arch == 'itransformer':
#             self.lstm = iTransformerEncoderClassiregressor(
#                         feat_dim=lstm_n_features,
#                         max_len=61,
#                         d_model=512,
#                         n_heads=8,
#                         num_layers=6,
#                         dim_feedforward=2048, 
#                         num_classes=lstm_out,
#                         dropout=0.1,
#                         pos_encoding="fixed",
#                         activation="gelu",
#                         norm="BatchNorm",
#                         freeze=False,
#                         )
#         else:
#             raise ValueError("Invalid RNN Architecture.")

#         # 光谱数据处理网络
#         if spectral_net_type == "vgg":
#             self.spectral_net = VGGSpectraNet(spectral_in_features, hidden_size)
#         elif spectral_net_type == "resnet":
#             self.spectral_net = ResSpectraNet(spectral_in_features, hidden_size)
#         elif spectral_net_type == "densenet":
#             self.spectral_net = DenseSpectraNet(spectral_in_features, hidden_size)
#         elif spectral_net_type == 'lstm':
#             self.spectral_net = LSTM(spectral_in_features, hidden_size)
#         elif spectral_net_type == "lstmcnn":
#             self.spectral_net = LSTMCNNAttentionNet(spectral_in_features, hidden_size)
#         elif spectral_net_type == "PLSR":
#             self.spectral_net = PLSRNet(spectral_in_features,hidden_size)
#         elif spectral_net_type == "svr":
#             self.spectral_net = SVM(spectral_in_features,hidden_size)
#         elif spectral_net_type == "mambacnn":
#             self.spectral_net = MambaCNNAttentionNet(spectral_in_features,hidden_size)
#         elif spectral_net_type == "rf":
#             self.spectral_net = RFSpectraNet(spectral_in_features,hidden_size)
#         elif spectral_net_type == "cnn":
#             self.spectral_net = CNNNet(spectral_in_features,hidden_size)
#         elif spectral_net_type == "lssvm":
#             self.spectral_net = LSSVM(spectral_in_features,hidden_size)
#         elif spectral_net_type == "cnnatten":
#             self.spectral_net = CNNNetWithAttention(spectral_in_features,hidden_size)
#         else:
#             # 默认简单MLP
#             self.spectral_net = nn.Sequential(
#                 nn.Linear(spectral_in_features, spectral_hidden),
#                 nn.ReLU(),
#                 nn.Dropout(0.3),
#                 nn.Linear(spectral_hidden, hidden_size),
#                 nn.ReLU()
#             )

#         # 修改回归器以处理三种特征，并传入权重参数
#         self.reg = MultiHeadRegressor(
#             regresor_input_from_cnn,  # CNN特征
#             lstm_out,                 # LSTM特征
#             hidden_size,              # 光谱特征
#             hidden_size=hidden_size,
#             version=reg_version,
#             # 传入权重参数
#             spectral_weight=spectral_weight,
#             image_weight=image_weight,
#             climate_weight=climate_weight
#         )
        
#     def forward(self, input_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
#         """
#         处理三种输入数据：遥感图像、时序气候数据和光谱数据
        
#         参数:
#         ------
#         input_data: 包含三个张量的元组:
#             * raster_stack (torch.Tensor): 形状为 (batch_size, channels, height, width) 的遥感图像
#             * ts_features (torch.Tensor): 形状为 (batch_size, seq_length, n_features) 的时序特征
#             * spectral_data (torch.Tensor): 形状为 (batch_size, spectral_features) 的光谱数据
            
#         返回:
#         -------
#             output (torch.Tensor): 形状为 (batch_size, 1) 的预测结果
#         """
#         raster_stack, ts_features, spectral_data = input_data
        
#         # 处理遥感图像
#         flat_raster = self.cnn(raster_stack)
        
#         # 处理时序数据
#         lstm_output = self.lstm(ts_features)
        
#         # 处理光谱数据
#         spectral_features = self.spectral_net(spectral_data)
        
#         # 融合三种特征并进行预测
#         # 权重已经在MultiHeadRegressor中应用
#         output = self.reg(flat_raster, lstm_output, spectral_features)
        
#         return output
class SoilNetJustCORR(SoilNetCORR):
    """
    This class inherits from SoilNetCORR but disables both the CNN and LSTM pathways
    to use only the spectral (CORR) data.
    """
    def __init__(self, use_glam=False, cnn_arch="resnet101", reg_version=1,
                 cnn_in_channels=14, regresor_input_from_cnn=1024,
                 lstm_n_features=10, lstm_n_layers=2, lstm_out=128,
                 spectral_in_features=2000, spectral_hidden=512,
                 hidden_size=128, rnn_arch="LSTM", seq_len=61, img_size=64,
                 spectral_net_type='vgg'):
        
        super().__init__(use_glam=use_glam, cnn_arch=cnn_arch, reg_version=reg_version,
                        cnn_in_channels=cnn_in_channels, regresor_input_from_cnn=regresor_input_from_cnn,
                        lstm_n_features=lstm_n_features, lstm_n_layers=lstm_n_layers, 
                        lstm_out=lstm_out, spectral_in_features=spectral_in_features,
                        spectral_hidden=spectral_hidden, hidden_size=hidden_size,
                        rnn_arch=rnn_arch, seq_len=seq_len, img_size=img_size,
                        spectral_net_type=spectral_net_type)
        
        # 禁用 CNN 和 LSTM 路径
        self.cnn = None
        self.lstm = None
        
        # 只保留光谱网络
        if spectral_net_type == "vgg":
            self.spectral_net = VGGSpectraNet(spectral_in_features, hidden_size)
        elif spectral_net_type == "resnet":
            self.spectral_net = ResSpectraNet(spectral_in_features, hidden_size)
        elif spectral_net_type == "densenet":
            self.spectral_net = DenseSpectraNet(spectral_in_features, hidden_size)
        else:
            self.spectral_net = nn.Sequential(
                nn.Linear(spectral_in_features, spectral_hidden),
                nn.ReLU(),
                nn.BatchNorm1d(spectral_hidden),
                nn.Dropout(0.3),
                nn.Linear(spectral_hidden, hidden_size),
                nn.ReLU()
            )
        
        # 修改回归器以只处理光谱特征
        self.reg = MultiHeadRegressor(hidden_size, hidden_size=hidden_size, version=reg_version)
    
    def forward(self, input_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        只处理光谱数据，忽略其他输入
        
        参数:
        ------
        input_data: 虽然输入仍为三个张量的元组，但只使用光谱数据:
            * raster_stack (torch.Tensor): 未使用
            * ts_features (torch.Tensor): 未使用
            * spectral_data (torch.Tensor): 形状为 (batch_size, spectral_features) 的光谱数据
            
        返回:
        -------
            output (torch.Tensor): 形状为 (batch_size, 1) 的预测结果
        """
        # 只使用第三个输入（光谱数据）
        _, _, spectral_data = input_data
        
        # 确保输入维度正确
        if len(spectral_data.shape) == 1:
            spectral_data = spectral_data.unsqueeze(0)
        
        # 处理光谱数据
        spectral_features = self.spectral_net(spectral_data)
        
        # 直接使用光谱特征进行预测
        output = self.reg(spectral_features)
        
        return output

            

 
class SoilNetSimCLR(nn.Module):
    def __init__(self, use_glam = False  , cnn_arch = "resnet101", reg_version = 1,
                 cnn_in_channels = 14 ,regresor_input_from_cnn = 128, 
                 lstm_n_features = 10,lstm_n_layers =2, lstm_out = 128, hidden_size=128, rnn_arch = "LSTM", seq_len = 61, img_size = 64):
        
        super().__init__()
        
        if use_glam:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                raise ValueError("ViT is not supported when GLAM is enabled. Please choose from 'resnet' or 'vgg16' or disable GLAM.")
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16'.")

        else:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            if cnn_arch == "resnet50":
                self.cnn = ResNet50(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                self.cnn = ViT(img_size=img_size, patch_size=8, in_chans=cnn_in_channels, n_classes=regresor_input_from_cnn, p=0.1, attn_p=0.1)
            else:
                raise ValueError("Invalid CNN Architecture. Please choose from 'resnet' or 'vgg16'.")
            

        if rnn_arch == "LSTM":
            self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "GRU":
            self.lstm = rnn.GRU(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "RNN":
            self.lstm = rnn.RNN(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "Transformer":
            self.lstm = TSTransformerEncoderClassiregressor(
                        feat_dim=lstm_n_features,
                        max_len=61,
                        d_model=512,
                        n_heads=8,
                        num_layers=6,
                        dim_feedforward=2048, 
                        num_classes=lstm_out,
                        dropout=0.1,
                        pos_encoding="fixed",
                        activation="gelu",
                        norm="BatchNorm",
                        freeze=False,
                        )
        else:
            raise ValueError("Invalid RNN Architecture. Please choose from 'LSTM', 'GRU' or 'RNN'.")
        
        #self.reg = MultiHeadRegressor(regresor_input_from_cnn, lstm_out, hidden_size= hidden_size, version=reg_version)
        
    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Inputs
        ------
        input_raster_ts : A tupple containing the following two tensors:
            * raster_stack (torch.Tensor): A 4D tensor of shape `(batch_size, channels, height, width)` representing a stack of raster images.
            * ts_features (torch.Tensor): A 3D tensor of shape `(batch_size, seq_length, , n_features)` representing a sequence of time-series features. | `seq_length` is the number of time steps in the sequence. e.g. months in our climate data
            
        Outputs
        -------
            - output (torch.Tensor, torch.Tensor): A tupples of tensors of shape `(batch_size, 128)` representing the embeddings of image and climate data.
            to be used in SimCLR loss.
        """
        raster_stack, ts_features = input_raster_ts
        flat_raster = self.cnn(raster_stack)
        lstm_output = self.lstm(ts_features)

        return flat_raster, lstm_output
                
    
class SoilNetSimCLRwRegHead(nn.Module):
    def __init__(self, soilnet_simclr: SoilNetSimCLR, hidden_size=128, reg_version = 1):
        super().__init__()
        self.soilnet_simclr = soilnet_simclr
        self.reg = MultiHeadRegressor(hidden_size, hidden_size, hidden_size= hidden_size, version=reg_version)
    def forward(self, input_raster_ts: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        Inputs
        ------
        input_raster_ts : A tupple containing the following two tensors:
            * raster_stack (torch.Tensor): A 4D tensor of shape `(batch_size, channels, height, width)` representing a stack of raster images.
            * ts_features (torch.Tensor): A 3D tensor of shape `(batch_size, seq_length, , n_features)` representing a sequence of time-series features. | `seq_length` is the number of time steps in the sequence. e.g. months in our climate data
            
        Outputs
        -------
            - output (torch.Tensor): A tensor of shape `(batch_size, 1)` representing the predicted output of regression.
        """
        raster_stack, ts_features = input_raster_ts
        flat_raster, lstm_output = self.soilnet_simclr((raster_stack, ts_features))
        output = self.reg(flat_raster, lstm_output)
        return output
class SoilNetSimCLR_CORR(nn.Module):
    def __init__(self, use_glam=False, cnn_arch="resnet101", reg_version=1,
                 cnn_in_channels=14, regresor_input_from_cnn=128, 
                 lstm_n_features=10, lstm_n_layers=2, lstm_out=128, 
                 spectral_in_features=2000, spectral_hidden=512,
                 hidden_size=128, rnn_arch="LSTM", seq_len=61, img_size=64,
                 spectral_net_type='vgg', projection_dim=128): 
        
        super().__init__()
        
        # CNN 部分保持不变
        if use_glam:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                raise ValueError("ViT is not supported when GLAM is enabled.")
            else:
                raise ValueError("Invalid CNN Architecture.")
        else:
            if cnn_arch == "resnet101":
                self.cnn = ResNet101(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            if cnn_arch == "resnet50":
                self.cnn = ResNet50(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "vgg16":
                self.cnn = VGG16GLAM(in_channels=cnn_in_channels, out_nodes=regresor_input_from_cnn)
            elif cnn_arch == "ViT":
                self.cnn = ViT(img_size=img_size, patch_size=8, in_chans=cnn_in_channels, 
                             n_classes=regresor_input_from_cnn, p=0.1, attn_p=0.1)
            else:
                raise ValueError("Invalid CNN Architecture.")

        # LSTM 部分保持不变
        if rnn_arch == "LSTM":
            self.lstm = rnn.LSTM(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "GRU":
            self.lstm = rnn.GRU(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "RNN":
            self.lstm = rnn.RNN(lstm_n_features, hidden_size, lstm_n_layers, lstm_out)
        elif rnn_arch == "Transformer":
            self.lstm = TSTransformerEncoderClassiregressor(
                        feat_dim=lstm_n_features,
                        max_len=61,
                        d_model=512,
                        n_heads=8,
                        num_layers=6,
                        dim_feedforward=2048, 
                        num_classes=lstm_out,
                        dropout=0.1,
                        pos_encoding="fixed",
                        activation="gelu",
                        norm="BatchNorm",
                        freeze=False,
                        )
        elif rnn_arch == "itransformer":
            self.lstm = iTransformerEncoderClassiregressor(
                        feat_dim=lstm_n_features,
                        max_len=61,
                        d_model=512,
                        n_heads=8,
                        num_layers=6,
                        dim_feedforward=2048, 
                        num_classes=lstm_out,
                        dropout=0.1,
                        pos_encoding="fixed",
                        activation="gelu",
                        norm="BatchNorm",
                        freeze=False,
                        )
        else:
            raise ValueError("Invalid RNN Architecture.")
            
        # 添加光谱数据处理网络
        if spectral_net_type == "vgg":
            self.spectral_net = VGGSpectraNet(spectral_in_features, hidden_size)
        elif spectral_net_type == "resnet":
            self.spectral_net = ResSpectraNet(spectral_in_features, hidden_size)
        elif spectral_net_type == "densenet":
            self.spectral_net = DenseSpectraNet(spectral_in_features, hidden_size)
        elif spectral_net_type == "lstmcnn":
            self.spectral_net = LSTMCNNAttentionNet(spectral_in_features, hidden_size)
        elif spectral_net_type == 'LSTM':
            self.spectral_net = LSTM(spectral_in_features, hidden_size,lstm_n_layers, lstm_out)
        elif spectral_net_type == "PLSR":
            self.spectral_net = PLSRNet(spectral_in_features,hidden_size)
        else:
            self.spectral_net = nn.Sequential(
                nn.Linear(spectral_in_features, spectral_hidden),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(spectral_hidden, hidden_size),
                nn.ReLU()
            )
        # 添加投影头
        self.img_projector = nn.Sequential(
            nn.Linear(regresor_input_from_cnn, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )
        
        self.clim_projector = nn.Sequential(
            nn.Linear(lstm_out, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )
        
        self.spec_projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, projection_dim)
        )
    def forward(self, input_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Inputs
        ------
        input_data : A tuple containing three tensors:
            * raster_stack (torch.Tensor): Shape (batch_size, channels, height, width)
            * ts_features (torch.Tensor): Shape (batch_size, seq_length, n_features)
            * spectral_data (torch.Tensor): Shape (batch_size, spectral_features)
            
        Returns
        -------
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
                - Normalized image embeddings (batch_size, projection_dim)
                - Normalized climate embeddings (batch_size, projection_dim)
                - Normalized spectral embeddings (batch_size, projection_dim)
        """
        raster_stack, ts_features, spectral_data = input_data
        
        # 获取特征
        img_features = self.cnn(raster_stack)
        clim_features = self.lstm(ts_features)
        spec_features = self.spectral_net(spectral_data)
        
        # 通过投影头
        z_img = self.img_projector(img_features)
        z_clim = self.clim_projector(clim_features)
        z_spec = self.spec_projector(spec_features)
        
        # L2 归一化
        z_img = F.normalize(z_img, dim=1)
        z_clim = F.normalize(z_clim, dim=1)
        z_spec = F.normalize(z_spec, dim=1)
        
        return z_img, z_clim, z_spec

class SoilNetSimCLRwRegHead_CORR(nn.Module):
    def __init__(self, soilnet_simclr: SoilNetSimCLR_CORR, hidden_size=128, reg_version=1):
        super().__init__()
        self.soilnet_simclr = soilnet_simclr
        self.reg = MultiHeadRegressor(hidden_size, hidden_size, hidden_size, 
                                    hidden_size=hidden_size, version=reg_version)
        
    def forward(self, input_data: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """
        用于微调阶段的前向传播
        """
        raster_stack, ts_features, spectral_data = input_data
        z_img, z_clim, z_spec = self.soilnet_simclr((raster_stack, ts_features, spectral_data))
        output = self.reg(z_img, z_clim, z_spec)
        return output

    
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    # print("Testing SoilNet...")
    # x = torch.randn((32,12,128,128))
    # y = torch.rand((32,12))
    # model = SoilNet()
    # z = model(x,y)
    # print(z.detach().shape)
    # print('Testing SoilNetFC...')
    # x = torch.randn((32,12,64,64))
    # model = SoilNetFC(cnn_in_channels=12)
    # y = model(x)
    # print(y.detach().shape)
    
    # print("Testing SoilNetMonoLSTM...")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Check if GPU is available
    # x_cnn = torch.randn((32,12,64,64)).to(device)
    # x_lstm = torch.randn((32, 12, 10)).to(device)
    # model = SoilNetMonoLSTM(cnn_in_channels=12, lstm_n_features=10).to(device)
    # y= model(x_cnn, x_lstm)
    # print(y.detach().shape)
    
    print("Testing SoilNet...")
    x = torch.randn((32,12,64,64))
    model = SoilNet(cnn_in_channels=12, cnn_arch="ViT")
    y = model(x)
    print(y.detach().shape)
    
    print('Testing SoilNetLSTM...')
    x_cnn = torch.randn((32,12,64,64)).to(device)
    x_lstm = torch.randn((32, 60, 10)).to(device)
    model = SoilNetLSTM(cnn_arch="ViT", cnn_in_channels= 12, regresor_input_from_cnn=1024,
                       lstm_n_features= 10, lstm_n_layers=2, lstm_out=128, hidden_size=128).to(device)
    y= model((x_cnn, x_lstm))
    print(y.detach().shape)
    
    print("Testing SoilNetSimCLR...")
    modelSimCLR = SoilNetSimCLR(cnn_arch="ViT", cnn_in_channels= 12, regresor_input_from_cnn=128,
                       lstm_n_features= 10, lstm_n_layers=2, lstm_out=128, hidden_size=128).to(device)
    y1, y2 = modelSimCLR((x_cnn, x_lstm))
    print(y1.detach().shape, y2.detach().shape)
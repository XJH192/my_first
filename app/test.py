# from train_utils import *
# import torch
# from torch.utils.data import DataLoader
# from torchvision import transforms
# import numpy as np
# from dataset.utils.utils import TextColors as tc
# from plot_utils.plot import plot_train_test_losses
# from datetime import date, datetime
# import torch.nn.functional as F
# import cv2
# import json
# import warnings
# import config
# from soilnet.soil_net import SoilNet, SoilNetLSTM, SoilNetSimCLRwRegHead, SoilNetJustLSTM,SoilNetCORR,SoilNetSimCLRwRegHead_CORR,SoilNetJustCORR
# import csv
# import train_utils
# from train_utils import *
# from datetime import date, datetime
# import argparse
# import pandas as pd
# import os
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from dataset.dataset_loader import SNDatasetCorr
# # Format the date and time
# now = datetime.now()
# start_string = now.strftime("%Y-%m-%d %H:%M:%S")
# device = "cuda" if torch.cuda.is_available() else "cpu"

# train_l8_folder_path = config.train_l8_folder_path
# test_l8_folder_path = config.test_l8_folder_path
# val_l8_folder_path = config.val_l8_folder_path
# lucas_csv_path = config.lucas_csv_path
# climate_csv_folder_path = config.climate_csv_folder_path
# spectral_csv_path=config.spectral_csv_path
# SIMCLR_PATH = config.SIMCLR_PATH

# EXP_NAME = 'LUCAS_Transformer_NoImage'
# DATASET = 'LUCAS'  # 'LUCAS', 'RaCA'
# NUM_WORKERS = 2
# TRAIN_BATCH_SIZE = 4
# TEST_BATCH_SIZE = 4
# LEARNING_RATE = 1e-3
# NUM_EPOCHS = 2
# LR_SCHEDULER = "step"  # step, plateau or None
# USE_SRTM = False
# USE_SPATIAL_ATTENTION = False
# CNN_ARCHITECTURE = "ViT"  # vgg16 or resnet101 or "ViT" or resnet50
# RNN_ARCHITECTURE = 'Transformer'  # LSTM, GRU, RNN, Transformer
# REG_VERSION = 1
# SEEDS = [1, ]  # Seeds for cross-validation and reproducibility
# USE_LSTM_BRANCH = False
# LOG_LOSS = False
# SAVE_TRAIN_DATA_METRICS = False
# LOAD_SIMCLR_MODEL = False
# JUST_LSTM = False  # Using Only Climate Data
# USE_SPECTRAL = False  # 是否使用光谱数据
# SPECTRAL_NET_TYPE = 'vgg'  # 光谱网络架构类型
# SPECTRAL_IN_FEATURES = 2000  # 光谱输入特征数
# SPECTRAL_HIDDEN = 512  # 光谱网络隐藏层维度
# def custom_collate_fn(batch):
#     """
#     自定义的collate函数来处理不同大小的数据
#     """
#     # 过滤掉空张量
#     batch = [item for item in batch if all(x.shape[0] > 0 for x in item[0])]
#     if len(batch) == 0:
#         raise RuntimeError("Empty batch after filtering")
    
#     # 分离数据
#     if len(batch[0]) == 3:  # 包含point_id
#         data_list = [item[0] for item in batch]
#         labels = [item[1] for item in batch]
#         ids = [item[2] for item in batch]
#     else:
#         data_list = [item[0] for item in batch]
#         labels = [item[1] for item in batch]
#         ids = None

#     # 处理每种输入数据
#     processed_data = []
#     for i in range(len(data_list[0])):  # 遍历每种输入类型
#         current_data = [d[i] for d in data_list]
#         try:
#             # 尝试直接堆叠
#             stacked_data = torch.stack(current_data)
#         except:
#             # 如果直接堆叠失败，进行维度对齐
#             if len(current_data[0].shape) == 2:  # 2D张量
#                 max_dim = max(x.shape[0] for x in current_data)
#                 padded_data = []
#                 for x in current_data:
#                     if x.shape[0] < max_dim:
#                         padding = torch.zeros((max_dim - x.shape[0], x.shape[1]), dtype=x.dtype)
#                         x = torch.cat([x, padding], dim=0)
#                     padded_data.append(x)
#                 stacked_data = torch.stack(padded_data)
#             else:
#                 # 处理其他维度的张量
#                 raise ValueError(f"Unsupported tensor dimension: {len(current_data[0].shape)}")
#         processed_data.append(stacked_data)

#     # 改进的标签处理方式
#     if torch.is_tensor(labels[0]):
#         labels = torch.stack([label.clone().detach() for label in labels])
#     else:
#         labels = torch.tensor(labels)

#     if ids is not None:
#         return processed_data, labels, ids
#     return processed_data, labels
# # def custom_collate_fn(batch):
# #     """
# #     自定义的collate函数来处理不同大小的数据
# #     """
# #     # 分离数据和标签
# #     data = [item[0] for item in batch]
# #     labels = [item[1] for item in batch]
    
# #     # 处理三种输入数据：图像、气候和光谱数据
# #     img_data = []
# #     climate_data = []
# #     spectral_data = []
    
# #     for d in data:
# #         img, clim, spec = d
# #         img_data.append(img)
# #         climate_data.append(clim)
# #         spectral_data.append(spec)
    
# #     # 处理图像数据
# #     try:
# #         img_batch = torch.stack(img_data)
# #     except:
# #         # 如果图像大小不一致，进行padding
# #         max_h = max([img.size(1) for img in img_data])
# #         max_w = max([img.size(2) for img in img_data])
# #         padded_imgs = []
# #         for img in img_data:
# #             pad_h = max_h - img.size(1)
# #             pad_w = max_w - img.size(2)
# #             padded = F.pad(img, (0, pad_w, 0, pad_h))
# #             padded_imgs.append(padded)
# #         img_batch = torch.stack(padded_imgs)
    
# #     # 处理气候数据
# #     try:
# #         climate_batch = torch.stack(climate_data)
# #     except:
# #         # 处理气候数据的不一致
# #         max_len = max([clim.size(0) for clim in climate_data])
# #         padded_clim = []
# #         for clim in climate_data:
# #             if clim.size(0) < max_len:
# #                 pad_size = max_len - clim.size(0)
# #                 padded = F.pad(clim, (0, 0, 0, pad_size))
# #                 padded_clim.append(padded)
# #             else:
# #                 padded_clim.append(clim)
# #         climate_batch = torch.stack(padded_clim)
    
# #     # 处理光谱数据
# #     try:
# #         spectral_batch = torch.stack(spectral_data)
# #     except:
# #         # 处理光谱数据的不一致
# #         max_len = max([spec.size(0) for spec in spectral_data])
# #         padded_spec = []
# #         for spec in spectral_data:
# #             if spec.size(0) < max_len:
# #                 pad_size = max_len - spec.size(0)
# #                 padded = F.pad(spec, (0, 0, 0, pad_size))
# #                 padded_spec.append(padded)
# #             else:
# #                 padded_spec.append(spec)
# #         spectral_batch = torch.stack(padded_spec)
    
# #     # 处理标签
# #     if torch.is_tensor(labels[0]):
# #         labels = torch.stack(labels)
# #     else:
# #         labels = torch.tensor(labels)
    
# #     return (img_batch, climate_batch, spectral_batch), labels


# def parse_arguments():
#     parser = argparse.ArgumentParser(description='SoilNet Testing')
#     parser.add_argument('-e', '--exp_name', type=str, default=EXP_NAME, help='Experiment name - helps to identify the experiment')
#     parser.add_argument('-d', '--dataset', type=str, default=DATASET, choices=['LUCAS', 'RaCA'], help='Dataset name to use')
#     parser.add_argument('-w', '--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for data loading')
#     parser.add_argument('-trbs', '--train_batch_size', type=int, default=TRAIN_BATCH_SIZE, help='Batch size for training')
#     parser.add_argument('-tsbs', '--test_batch_size', type=int, default=TEST_BATCH_SIZE, help='Batch size for testing')
#     parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
#     parser.add_argument('-ne', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
#     parser.add_argument('-ls', '--lr_scheduler', type=str, default=LR_SCHEDULER, choices=['step', 'plateau', 'None'], help='Learning rate scheduler')
#     parser.add_argument('-srtm', '--use_srtm', action='store_true', default=USE_SRTM, help='Use SRTM data')
#     parser.add_argument('-sa', '--use_spatial_attention', action='store_true', default=USE_SPATIAL_ATTENTION, help='Use spatial attention')
#     parser.add_argument('-cnn', '--cnn_architecture', type=str, default=CNN_ARCHITECTURE, choices=['vgg16', 'resnet101', 'ViT', 'resnet50','ViM','MiV'], help='CNN architecture')
#     parser.add_argument('-rnn', '--rnn_architecture', type=str, default=RNN_ARCHITECTURE, choices=['LSTM', 'GRU', 'RNN', 'Transformer','mamba','itransformer'], help='RNN architecture')
#     parser.add_argument('-rv', '--reg_version', type=int, default=REG_VERSION, help='Regression version')
#     parser.add_argument('-s', '--seeds', nargs='+', type=int, default=SEEDS, help='Seeds for cross-validation. input example: 1 2 3 4 5')
#     parser.add_argument('-lstm', '--use_lstm_branch', action='store_true', default=USE_LSTM_BRANCH, help='Use Climate data - I know! the name is misleading')
#     parser.add_argument('-log', '--log_loss', action='store_true', default=LOG_LOSS, help='Use logarithmic loss')
#     parser.add_argument('-stm', '--save_train_data_metrics', action='store_true', default=SAVE_TRAIN_DATA_METRICS, help='Save training data metrics')
#     parser.add_argument('-simclr', '--load_simclr_model', action='store_true', default=LOAD_SIMCLR_MODEL, help='Load Self-supervised model to fine-tune')
#     parser.add_argument('-jlstm', '--just_lstm', action='store_true', default=JUST_LSTM, help='Use only climate data')
#     parser.add_argument('-spec', '--use_spectral', action='store_true', default=USE_SPECTRAL, help='Use spectral data')
#     parser.add_argument('-snt', '--spectral_net_type', type=str, default=SPECTRAL_NET_TYPE,choices=['vgg', 'resnet', 'densenet','lstmcnn','mambacnn','svr','cnnatten','cnn','lstm'], help='Spectral network architecture type')
#     parser.add_argument('-sf', '--spectral_in_features', type=int, default=SPECTRAL_IN_FEATURES,help='Number of input features for spectral data')
#     parser.add_argument('-sh', '--spectral_hidden', type=int, default=SPECTRAL_HIDDEN,help='Hidden dimension for spectral network')
#     parser.add_argument('-jcorr', '--just_corr', action='store_true', default=False, help='Use only spectral data')
#     args = parser.parse_args()
#     return args


# def evaluate_regression_metrics(y_true, y_pred):
#     """
#     计算回归任务的评价指标。
#     :param y_true: 真实值 (numpy array 或 list)
#     :param y_pred: 预测值 (numpy array 或 list)
#     :return: RMSE, R2, RPIQ, MAE, MEC, CCC
#     """
#     # 计算 RMSE
#     rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
#     # 计算 R²
#     r2 = r2_score(y_true, y_pred)
    
#     # 计算 MAE
#     mae = mean_absolute_error(y_true, y_pred)
    
#     # 计算 MEC (Model Efficiency Coefficient)
#     mec = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
#     # 计算 CCC (Concordance Correlation Coefficient)
#     mean_true = np.mean(y_true)
#     mean_pred = np.mean(y_pred)
#     var_true = np.var(y_true)
#     var_pred = np.var(y_pred)
#     covar = np.cov(y_true, y_pred)[0, 1]
#     ccc = (2 * covar) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    
#     # 计算 RPIQ (Ratio of Performance to Interquartile Range)
#     iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
#     rpiq = iqr / rmse
    
#     return rmse, r2, rpiq, mae, mec, ccc


# def test_step_w_id(model, data_loader, loss_fn, verbose=False, csv_file=None):
#     model.eval()
#     with torch.no_grad():
#         # 如果是第一次写入，添加表头
#         if csv_file and not os.path.exists(csv_file):
#             with open(csv_file, mode='w', newline='') as file:
#                 writer = csv.writer(file)
#                 writer.writerow(['point_id', 'y_real', 'y_pred'])  # 写入表头

#         for i, batch_data in enumerate(data_loader):
#             # 解包数据
#             if len(batch_data) == 3:  # 包含point_id的情况
#                 X_list, y, id_tuple = batch_data
#             else:
#                 X_list, y = batch_data
#                 id_tuple = None

#             # 将数据移到设备上
#             X_list = [x.to(device) for x in X_list]
#             y = y.to(device)

#             # 处理时间序列特征（如果存在）
#             if len(X_list) >= 2:  # 如果有时间序列数据
#                 ts_features = X_list[1]
#                 print(f"Original time series features shape: {ts_features.shape}")
                
#                 # 调整维度为 (batch_size, 61, 13)
#                 if len(ts_features.shape) > 3:  # 如果需要降维
#                     ts_features = ts_features.permute(0, 4, 1, 2, 3)
#                     ts_features = ts_features.mean(dim=[-2, -1])
                
#                 print(f"Adjusted time series features shape: {ts_features.shape}")
#                 X_list[1] = ts_features

#             # 处理光谱数据（如果存在）
#             if len(X_list) >= 3:  # 如果有光谱数据
#                 spectral_features = X_list[2]
#                 print(f"Spectral features shape: {spectral_features.shape}")
#                 # 确保光谱数据形状正确，如果需要可以在这里添加处理逻辑

#             # 将调整后的输入数据传递给模型
#             y_pred = model(X_list)

#             # 计算损失
#             loss = loss_fn(y_pred, y)

#             # 打印结果
#             if verbose:
#                 print(f"Batch {i}: Loss = {loss.item()}")

#             # 保存结果到 CSV 文件
#             if csv_file and id_tuple is not None:
#                 with open(csv_file, mode='a', newline='') as file:
#                     writer = csv.writer(file)
#                     for j in range(len(y)):
#                         writer.writerow([id_tuple[j], y[j].item(), y_pred[j].item()])

# if __name__ == '__main__':
#     args = parse_arguments()
#     EXP_NAME = args.exp_name
#     DATASET = args.dataset
#     NUM_WORKERS = args.num_workers
#     TRAIN_BATCH_SIZE = args.train_batch_size
#     TEST_BATCH_SIZE = args.test_batch_size
#     LEARNING_RATE = args.learning_rate
#     NUM_EPOCHS = args.num_epochs
#     LR_SCHEDULER = args.lr_scheduler
#     USE_SRTM = args.use_srtm
#     USE_SPATIAL_ATTENTION = args.use_spatial_attention
#     CNN_ARCHITECTURE = args.cnn_architecture
#     RNN_ARCHITECTURE = args.rnn_architecture
#     REG_VERSION = args.reg_version
#     SEEDS = args.seeds
#     USE_LSTM_BRANCH = args.use_lstm_branch
#     LOG_LOSS = args.log_loss
#     SAVE_TRAIN_DATA_METRICS = args.save_train_data_metrics
#     LOAD_SIMCLR_MODEL = args.load_simclr_model
#     JUST_LSTM = args.just_lstm
# 	# 在args解析后添加
#     USE_SPECTRAL = args.use_spectral
#     SPECTRAL_NET_TYPE = args.spectral_net_type
#     SPECTRAL_IN_FEATURES = args.spectral_in_features
#     SPECTRAL_HIDDEN = args.spectral_hidden
#     JUST_CORR = args.just_corr
#     if JUST_CORR:
#         USE_SPECTRAL = True
#         USE_LSTM_BRANCH = False
#         JUST_LSTM = False
  
#     if DATASET == 'LUCAS':
#         from dataset.dataset_loader import SNDataset, SNDatasetClimate, myNormalize, myToTensor, Augmentations
#         OC_MAX = 560.2
#     if DATASET == 'RaCA':
#         from dataset.dataset_loader_us import SNDataset, SNDatasetClimate, myNormalize, myToTensor, Augmentations
#         OC_MAX = 4115

#     if USE_SRTM:
#         mynorm = myNormalize(img_bands_min_max=[[(0, 7), (0, 1)], [(7, 12), (-1, 1)], [(12), (-4, 2963)], [(13), (0, 90)]], oc_min=0, oc_max=OC_MAX)
#     else:
#         mynorm = myNormalize(img_bands_min_max=[[(0, 7), (0, 1)], [(7, 12), (-1, 1)]], oc_min=0, oc_max=OC_MAX)

#     my_to_tensor = myToTensor()
#     my_augmentation = Augmentations()
#     train_transform = transforms.Compose([mynorm, my_to_tensor, my_augmentation])
#     test_transform = transforms.Compose([mynorm, my_to_tensor])

#     bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] if not USE_SRTM else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
#     # 首先修改数据集选择部分
#     if not USE_LSTM_BRANCH:  # NOT USING THE CLIMATE DATA
#         test_ds_w_id = SNDataset(test_l8_folder_path, lucas_csv_path, l8_bands=bands, 
#                                 transform=test_transform, return_point_id=True)
#     elif USE_SPECTRAL:  # USING THE SPECTRAL DATA
#         test_ds_w_id = SNDatasetCorr(test_l8_folder_path, lucas_csv_path, 
#                                     climate_csv_folder_path, spectral_csv_path,
#                                     l8_bands=bands, transform=test_transform, 
#                                     return_point_id=True)
#     else:  # USING THE CLIMATE DATA ONLY
#         test_ds_w_id = SNDatasetClimate(test_l8_folder_path, lucas_csv_path, 
#                                     climate_csv_folder_path, l8_bands=bands, 
#                                     transform=test_transform, return_point_id=True)

#     SEQ_LEN = test_ds_w_id[0][0][1].shape[0]

#     CSV_FILES = [f for f in os.listdir(climate_csv_folder_path) if f.endswith('.csv')]
#     NUM_CLIMATE_FEATURES = len(CSV_FILES)

#     now = datetime.now()
#     run_name = now.strftime("D_%Y_%m_%d_T_%H_%M")
#     print("Current Date and Time:", run_name)
#     # 创建数据加载器

#     if not JUST_LSTM and not JUST_CORR:
#         if USE_LSTM_BRANCH and not USE_SPECTRAL:
#             model = SoilNetLSTM(use_glam=USE_SPATIAL_ATTENTION, cnn_arch=CNN_ARCHITECTURE, reg_version=REG_VERSION,
#                                 cnn_in_channels=len(bands), regresor_input_from_cnn=1024,#vim模型是128，其余是1024
#                                 lstm_n_features=NUM_CLIMATE_FEATURES, lstm_n_layers=2, lstm_out=128,
#                                 hidden_size=128, rnn_arch=RNN_ARCHITECTURE, seq_len=SEQ_LEN).to(device)
#         elif USE_SPECTRAL:
#                     model = SoilNetCORR(
#                     use_glam=USE_SPATIAL_ATTENTION,
#                     cnn_arch=CNN_ARCHITECTURE,
#                     reg_version=REG_VERSION,
#                     cnn_in_channels=len(bands),
#                     regresor_input_from_cnn=1024,
#                     lstm_n_features=NUM_CLIMATE_FEATURES,
#                     lstm_n_layers=2,
#                     lstm_out=128,
#                     spectral_in_features=2000,  # 新增光谱数据输入维度
#                     spectral_hidden=512,        # 新增光谱数据隐藏层维度
#                     hidden_size=128,
#                     rnn_arch=RNN_ARCHITECTURE,
#                     seq_len=SEQ_LEN,
#                     img_size=64,               # 默认图像大小
#                     spectral_net_type= SPECTRAL_NET_TYPE  # 默认使用VGG作为光谱网络
#                 ).to(device)
#         else:
#             model = SoilNet(use_glam=USE_SPATIAL_ATTENTION, cnn_arch=CNN_ARCHITECTURE, reg_version=REG_VERSION,
#                             cnn_in_channels=len(bands), regresor_input_from_cnn=1024, hidden_size=128).to(device)
#     elif JUST_LSTM:
#         model = SoilNetJustLSTM(use_glam=USE_SPATIAL_ATTENTION, cnn_arch=CNN_ARCHITECTURE, reg_version=REG_VERSION,
#                                 cnn_in_channels=len(bands), regresor_input_from_cnn=1024,
#                                 lstm_n_features=NUM_CLIMATE_FEATURES, lstm_n_layers=2, lstm_out=128,
#                                 hidden_size=128, rnn_arch=RNN_ARCHITECTURE, seq_len=SEQ_LEN).to(device)
#     else:
#         model = SoilNetJustCORR(
#            use_glam=USE_SPATIAL_ATTENTION,
#             cnn_arch=CNN_ARCHITECTURE,
#             reg_version=REG_VERSION,
#             cnn_in_channels=len(bands),
#             regresor_input_from_cnn=1024,
#             lstm_n_features=NUM_CLIMATE_FEATURES,
#             lstm_n_layers=2,
#             lstm_out=128,
#             spectral_in_features=SPECTRAL_IN_FEATURES,
#             spectral_hidden=SPECTRAL_HIDDEN,
#             hidden_size=128,
#             rnn_arch=RNN_ARCHITECTURE,
#             seq_len=SEQ_LEN,
#             img_size=64,
#             spectral_net_type=SPECTRAL_NET_TYPE
#         ).to(device)
           
#     if LOAD_SIMCLR_MODEL:
#         model = torch.load(SIMCLR_PATH).to(device)
#         if USE_SPECTRAL:
#             model=SoilNetSimCLRwRegHead_CORR(model,hidden_size=128,reg_version=REG_VERSION).to(device)
#         else:
#             model = SoilNetSimCLRwRegHead(model, hidden_size=128, reg_version=REG_VERSION).to(device)


#     # Load the best model
#     BEST_MODEL_PATH = "results/RUN_LUCAS_Transformer_NoImage_D_2025_07_08_T_16_57_best.pth.tar"
#     load_checkpoint(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), filename=BEST_MODEL_PATH)
#     model.eval()
#     print("Best Model loaded")
#     # 在调用模型之前，打印输入数据的结构

#     # 创建数据加载器
#     test_dl_w_id = DataLoader(test_ds_w_id, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,collate_fn=custom_collate_fn)
#     # test_dl_w_id = DataLoader(test_ds_w_id, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

#     # 调用 test_step_w_id
#     test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=True, csv_file=f"results/RUN_{EXP_NAME}_{run_name}_best.csv")
#     print(f"Best model saved to results/RUN_{EXP_NAME}_{run_name}_best.csv")

#     # Load the worst model
#     WORST_MODEL_PATH = "results/RUN_LUCAS_Transformer_NoImage_D_2025_07_08_T_20_10_worst.pth.tar"
#     load_checkpoint(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), filename=WORST_MODEL_PATH)
#     model.eval()
#     print("Worst Model loaded")

#     # 调用 test_step_w_id
#     test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=True, csv_file=f"results/RUN_{EXP_NAME}_{run_name}_worst.csv")
#     print(f"Worst Model saved to results/RUN_{EXP_NAME}_{run_name}_worst.csv")

#     # 计算最佳模型的评价指标
#     df = pd.read_csv(f"results/RUN_{EXP_NAME}_{run_name}_best.csv")
#     y_true = df['y_real'].values * OC_MAX
#     y_pred = df['y_pred'].values * OC_MAX

#     rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(y_true, y_pred)

#     best_dict = {
#         'RMSE': rmse,
#         'R2': r2,
#         'RPIQ': rpiq,
#         'MAE': mae,
#         'MEC': mec,
#         'CCC': ccc
#     }

#     # 计算最差模型的评价指标
#     df = pd.read_csv(f"results/RUN_{EXP_NAME}_{run_name}_worst.csv")
#     y_true = df['y_real'].values * OC_MAX
#     y_pred = df['y_pred'].values * OC_MAX

#     rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(y_true, y_pred)

#     worst_dict = {
#         'RMSE': rmse,
#         'R2': r2,
#         'RPIQ': rpiq,
#         'MAE': mae,
#         'MEC': mec,
#         'CCC': ccc
#     }

#     # 保存结果到 JSON 文件
#     cv_results_full = {
#         'best_dict': best_dict,
#         'worst_dict': worst_dict
#     }

#     with open(f"results/RUN_{EXP_NAME}_{run_name}.json", "w") as fp:
#         json.dump(cv_results_full, fp, indent=4)
from train_utils import *
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from dataset.utils.utils import TextColors as tc
from plot_utils.plot import plot_train_test_losses
from datetime import date, datetime
import torch.nn.functional as F
import cv2
import json
import warnings
import config
from soilnet.soil_net import SoilNet, SoilNetLSTM, SoilNetSimCLRwRegHead, SoilNetJustLSTM,SoilNetCORR,SoilNetSimCLRwRegHead_CORR,SoilNetJustCORR
import csv
import train_utils
from train_utils import *
from datetime import date, datetime
import argparse
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from dataset.dataset_loader import SNDatasetCorr
# Format the date and time
now = datetime.now()
start_string = now.strftime("%Y-%m-%d %H:%M:%S")
device = "cuda" if torch.cuda.is_available() else "cpu"

train_l8_folder_path = config.train_l8_folder_path
test_l8_folder_path = config.test_l8_folder_path
val_l8_folder_path = config.val_l8_folder_path
lucas_csv_path = config.lucas_csv_path
climate_csv_folder_path = config.climate_csv_folder_path
spectral_csv_path=config.spectral_csv_path
SIMCLR_PATH = config.SIMCLR_PATH

EXP_NAME = 'LUCAS_Transformer_NoImage'
DATASET = 'LUCAS'  # 'LUCAS', 'RaCA'
NUM_WORKERS = 2
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 4
LEARNING_RATE = 1e-3
NUM_EPOCHS = 2
LR_SCHEDULER = "step"  # step, plateau or None
USE_SRTM = True
USE_SPATIAL_ATTENTION = False
CNN_ARCHITECTURE = "ViT"  # vgg16 or resnet101 or "ViT" or resnet50
RNN_ARCHITECTURE = 'Transformer'  # LSTM, GRU, RNN, Transformer
REG_VERSION = 1
SEEDS = [1, ]  # Seeds for cross-validation and reproducibility
USE_LSTM_BRANCH = True
LOG_LOSS = False
SAVE_TRAIN_DATA_METRICS = False
LOAD_SIMCLR_MODEL = False
JUST_LSTM = False  # Using Only Climate Data
USE_SPECTRAL = True  # 是否使用光谱数据
SPECTRAL_NET_TYPE = 'lstmcnn'  # 光谱网络架构类型
SPECTRAL_IN_FEATURES = 2000  # 光谱输入特征数
SPECTRAL_HIDDEN = 512  # 光谱网络隐藏层维度
def custom_collate_fn(batch):
    """
    自定义的collate函数来处理不同大小的数据
    """
    # 过滤掉空张量
    batch = [item for item in batch if all(x.shape[0] > 0 for x in item[0])]
    if len(batch) == 0:
        raise RuntimeError("Empty batch after filtering")
    
    # 分离数据
    if len(batch[0]) == 3:  # 包含point_id
        data_list = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        ids = [item[2] for item in batch]
    else:
        data_list = [item[0] for item in batch]
        labels = [item[1] for item in batch]
        ids = None

    # 处理每种输入数据
    processed_data = []
    for i in range(len(data_list[0])):  # 遍历每种输入类型
        current_data = [d[i] for d in data_list]
        try:
            # 尝试直接堆叠
            stacked_data = torch.stack(current_data)
        except:
            # 如果直接堆叠失败，进行维度对齐
            if len(current_data[0].shape) == 2:  # 2D张量
                max_dim = max(x.shape[0] for x in current_data)
                padded_data = []
                for x in current_data:
                    if x.shape[0] < max_dim:
                        padding = torch.zeros((max_dim - x.shape[0], x.shape[1]), dtype=x.dtype)
                        x = torch.cat([x, padding], dim=0)
                    padded_data.append(x)
                stacked_data = torch.stack(padded_data)
            else:
                # 处理其他维度的张量
                raise ValueError(f"Unsupported tensor dimension: {len(current_data[0].shape)}")
        processed_data.append(stacked_data)

    # 改进的标签处理方式
    if torch.is_tensor(labels[0]):
        labels = torch.stack([label.clone().detach() for label in labels])
    else:
        labels = torch.tensor(labels)

    if ids is not None:
        return processed_data, labels, ids
    return processed_data, labels
# def custom_collate_fn(batch):
#     """
#     自定义的collate函数来处理不同大小的数据
#     """
#     # 分离数据和标签
#     data = [item[0] for item in batch]
#     labels = [item[1] for item in batch]
    
#     # 处理三种输入数据：图像、气候和光谱数据
#     img_data = []
#     climate_data = []
#     spectral_data = []
    
#     for d in data:
#         img, clim, spec = d
#         img_data.append(img)
#         climate_data.append(clim)
#         spectral_data.append(spec)
    
#     # 处理图像数据
#     try:
#         img_batch = torch.stack(img_data)
#     except:
#         # 如果图像大小不一致，进行padding
#         max_h = max([img.size(1) for img in img_data])
#         max_w = max([img.size(2) for img in img_data])
#         padded_imgs = []
#         for img in img_data:
#             pad_h = max_h - img.size(1)
#             pad_w = max_w - img.size(2)
#             padded = F.pad(img, (0, pad_w, 0, pad_h))
#             padded_imgs.append(padded)
#         img_batch = torch.stack(padded_imgs)
    
#     # 处理气候数据
#     try:
#         climate_batch = torch.stack(climate_data)
#     except:
#         # 处理气候数据的不一致
#         max_len = max([clim.size(0) for clim in climate_data])
#         padded_clim = []
#         for clim in climate_data:
#             if clim.size(0) < max_len:
#                 pad_size = max_len - clim.size(0)
#                 padded = F.pad(clim, (0, 0, 0, pad_size))
#                 padded_clim.append(padded)
#             else:
#                 padded_clim.append(clim)
#         climate_batch = torch.stack(padded_clim)
    
#     # 处理光谱数据
#     try:
#         spectral_batch = torch.stack(spectral_data)
#     except:
#         # 处理光谱数据的不一致
#         max_len = max([spec.size(0) for spec in spectral_data])
#         padded_spec = []
#         for spec in spectral_data:
#             if spec.size(0) < max_len:
#                 pad_size = max_len - spec.size(0)
#                 padded = F.pad(spec, (0, 0, 0, pad_size))
#                 padded_spec.append(padded)
#             else:
#                 padded_spec.append(spec)
#         spectral_batch = torch.stack(padded_spec)
    
#     # 处理标签
#     if torch.is_tensor(labels[0]):
#         labels = torch.stack(labels)
#     else:
#         labels = torch.tensor(labels)
    
#     return (img_batch, climate_batch, spectral_batch), labels


def parse_arguments():
    parser = argparse.ArgumentParser(description='SoilNet Testing')
    parser.add_argument('-e', '--exp_name', type=str, default=EXP_NAME, help='Experiment name - helps to identify the experiment')
    parser.add_argument('-d', '--dataset', type=str, default=DATASET, choices=['LUCAS', 'RaCA'], help='Dataset name to use')
    parser.add_argument('-w', '--num_workers', type=int, default=NUM_WORKERS, help='Number of workers for data loading')
    parser.add_argument('-trbs', '--train_batch_size', type=int, default=TRAIN_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('-tsbs', '--test_batch_size', type=int, default=TEST_BATCH_SIZE, help='Batch size for testing')
    parser.add_argument('-lr', '--learning_rate', type=float, default=LEARNING_RATE, help='Learning rate')
    parser.add_argument('-ne', '--num_epochs', type=int, default=NUM_EPOCHS, help='Number of epochs')
    parser.add_argument('-ls', '--lr_scheduler', type=str, default=LR_SCHEDULER, choices=['step', 'plateau', 'None'], help='Learning rate scheduler')
    parser.add_argument('-srtm', '--use_srtm', action='store_true', default=USE_SRTM, help='Use SRTM data')
    parser.add_argument('-sa', '--use_spatial_attention', action='store_true', default=USE_SPATIAL_ATTENTION, help='Use spatial attention')
    parser.add_argument('-cnn', '--cnn_architecture', type=str, default=CNN_ARCHITECTURE, choices=['vgg16', 'resnet101', 'ViT', 'resnet50','ViM','MiV'], help='CNN architecture')
    parser.add_argument('-rnn', '--rnn_architecture', type=str, default=RNN_ARCHITECTURE, choices=['LSTM', 'GRU', 'RNN', 'Transformer','mamba','itransformer'], help='RNN architecture')
    parser.add_argument('-rv', '--reg_version', type=int, default=REG_VERSION, help='Regression version')
    parser.add_argument('-s', '--seeds', nargs='+', type=int, default=SEEDS, help='Seeds for cross-validation. input example: 1 2 3 4 5')
    parser.add_argument('-lstm', '--use_lstm_branch', action='store_true', default=USE_LSTM_BRANCH, help='Use Climate data - I know! the name is misleading')
    parser.add_argument('-log', '--log_loss', action='store_true', default=LOG_LOSS, help='Use logarithmic loss')
    parser.add_argument('-stm', '--save_train_data_metrics', action='store_true', default=SAVE_TRAIN_DATA_METRICS, help='Save training data metrics')
    parser.add_argument('-simclr', '--load_simclr_model', action='store_true', default=LOAD_SIMCLR_MODEL, help='Load Self-supervised model to fine-tune')
    parser.add_argument('-jlstm', '--just_lstm', action='store_true', default=JUST_LSTM, help='Use only climate data')
    parser.add_argument('-spec', '--use_spectral', action='store_true', default=USE_SPECTRAL, help='Use spectral data')
    parser.add_argument('-snt', '--spectral_net_type', type=str, default=SPECTRAL_NET_TYPE,choices=['vgg', 'resnet', 'densenet','lstmcnn','mambacnn','svr','cnnatten','cnn','lstm'], help='Spectral network architecture type')
    parser.add_argument('-sf', '--spectral_in_features', type=int, default=SPECTRAL_IN_FEATURES,help='Number of input features for spectral data')
    parser.add_argument('-sh', '--spectral_hidden', type=int, default=SPECTRAL_HIDDEN,help='Hidden dimension for spectral network')
    parser.add_argument('-jcorr', '--just_corr', action='store_true', default=False, help='Use only spectral data')
    parser.add_argument('--spectral_weight', type=float, default=0.6, help='Weight for spectral data contribution (default: 0.6)')
    parser.add_argument('--image_weight', type=float, default=0.2, help='Weight for image data contribution (default: 0.2)')
    parser.add_argument('--climate_weight', type=float, default=0.2, help='Weight for climate data contribution (default: 0.2)')
    args = parser.parse_args()
    return args


def evaluate_regression_metrics(y_true, y_pred):
    """
    计算回归任务的评价指标。
    :param y_true: 真实值 (numpy array 或 list)
    :param y_pred: 预测值 (numpy array 或 list)
    :return: RMSE, R2, RPIQ, MAE, MEC, CCC
    """
    # 计算 RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # 计算 R²
    r2 = r2_score(y_true, y_pred)
    
    # 计算 MAE
    mae = mean_absolute_error(y_true, y_pred)
    
    # 计算 MEC (Model Efficiency Coefficient)
    mec = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    # 计算 CCC (Concordance Correlation Coefficient)
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    covar = np.cov(y_true, y_pred)[0, 1]
    ccc = (2 * covar) / (var_true + var_pred + (mean_true - mean_pred) ** 2)
    
    # 计算 RPIQ (Ratio of Performance to Interquartile Range)
    iqr = np.percentile(y_true, 75) - np.percentile(y_true, 25)
    rpiq = iqr / rmse
    
    return rmse, r2, rpiq, mae, mec, ccc


def test_step_w_id(model, data_loader, loss_fn, verbose=False, csv_file=None):
    model.eval()
    with torch.no_grad():
        # 如果是第一次写入，添加表头
        if csv_file and not os.path.exists(csv_file):
            with open(csv_file, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(['point_id', 'y_real', 'y_pred'])  # 写入表头

        for i, batch_data in enumerate(data_loader):
            # 解包数据
            if len(batch_data) == 3:  # 包含point_id的情况
                X_list, y, id_tuple = batch_data
            else:
                X_list, y = batch_data
                id_tuple = None

            # 将数据移到设备上
            X_list = [x.to(device) for x in X_list]
            y = y.to(device)

            # 处理时间序列特征（如果存在）
            if len(X_list) >= 2:  # 如果有时间序列数据
                ts_features = X_list[1]
                print(f"Original time series features shape: {ts_features.shape}")
                
                # 调整维度为 (batch_size, 61, 13)
                if len(ts_features.shape) > 3:  # 如果需要降维
                    ts_features = ts_features.permute(0, 4, 1, 2, 3)
                    ts_features = ts_features.mean(dim=[-2, -1])
                
                print(f"Adjusted time series features shape: {ts_features.shape}")
                X_list[1] = ts_features

            # 处理光谱数据（如果存在）
            if len(X_list) >= 3:  # 如果有光谱数据
                spectral_features = X_list[2]
                print(f"Spectral features shape: {spectral_features.shape}")
                # 确保光谱数据形状正确，如果需要可以在这里添加处理逻辑

            # 将调整后的输入数据传递给模型
            y_pred = model(X_list)

            # 计算损失
            loss = loss_fn(y_pred, y)

            # 打印结果
            if verbose:
                print(f"Batch {i}: Loss = {loss.item()}")

            # 保存结果到 CSV 文件
            if csv_file and id_tuple is not None:
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    for j in range(len(y)):
                        writer.writerow([id_tuple[j], y[j].item(), y_pred[j].item()])

if __name__ == '__main__':
    args = parse_arguments()
    EXP_NAME = args.exp_name
    DATASET = args.dataset
    NUM_WORKERS = args.num_workers
    TRAIN_BATCH_SIZE = args.train_batch_size
    TEST_BATCH_SIZE = args.test_batch_size
    LEARNING_RATE = args.learning_rate
    NUM_EPOCHS = args.num_epochs
    LR_SCHEDULER = args.lr_scheduler
    USE_SRTM = args.use_srtm
    USE_SPATIAL_ATTENTION = args.use_spatial_attention
    CNN_ARCHITECTURE = args.cnn_architecture
    RNN_ARCHITECTURE = args.rnn_architecture
    REG_VERSION = args.reg_version
    SEEDS = args.seeds
    USE_LSTM_BRANCH = args.use_lstm_branch
    LOG_LOSS = args.log_loss
    SAVE_TRAIN_DATA_METRICS = args.save_train_data_metrics
    LOAD_SIMCLR_MODEL = args.load_simclr_model
    JUST_LSTM = args.just_lstm
	# 在args解析后添加
    USE_SPECTRAL = args.use_spectral
    SPECTRAL_NET_TYPE = args.spectral_net_type
    SPECTRAL_IN_FEATURES = args.spectral_in_features
    SPECTRAL_HIDDEN = args.spectral_hidden
    JUST_CORR = args.just_corr
    SPECTRAL_WEIGHT = args.spectral_weight  # 新增
    IMAGE_WEIGHT = args.image_weight        # 新增
    CLIMATE_WEIGHT = args.climate_weight    # 新增
    if JUST_CORR:
        USE_SPECTRAL = True
        USE_LSTM_BRANCH = False
        JUST_LSTM = False
  
    if DATASET == 'LUCAS':
        from dataset.dataset_loader import SNDataset, SNDatasetClimate, myNormalize, myToTensor, Augmentations
        OC_MAX = 560.2
    if DATASET == 'RaCA':
        from dataset.dataset_loader_us import SNDataset, SNDatasetClimate, myNormalize, myToTensor, Augmentations
        OC_MAX = 4115

    if USE_SRTM:
        mynorm = myNormalize(img_bands_min_max=[[(0, 7), (0, 1)], [(7, 12), (-1, 1)], [(12), (-4, 2963)], [(13), (0, 90)]], oc_min=0, oc_max=OC_MAX)
    else:
        mynorm = myNormalize(img_bands_min_max=[[(0, 7), (0, 1)], [(7, 12), (-1, 1)]], oc_min=0, oc_max=OC_MAX)

    my_to_tensor = myToTensor()
    my_augmentation = Augmentations()
    train_transform = transforms.Compose([mynorm, my_to_tensor, my_augmentation])
    test_transform = transforms.Compose([mynorm, my_to_tensor])

    bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11] if not USE_SRTM else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    # 首先修改数据集选择部分
    if not USE_LSTM_BRANCH:  # NOT USING THE CLIMATE DATA
        test_ds_w_id = SNDataset(test_l8_folder_path, lucas_csv_path, l8_bands=bands, 
                                transform=test_transform, return_point_id=True)
    elif USE_SPECTRAL:  # USING THE SPECTRAL DATA
        test_ds_w_id = SNDatasetCorr(test_l8_folder_path, lucas_csv_path, 
                                    climate_csv_folder_path, spectral_csv_path,
                                    l8_bands=bands, transform=test_transform, 
                                    return_point_id=True)
    else:  # USING THE CLIMATE DATA ONLY
        test_ds_w_id = SNDatasetClimate(test_l8_folder_path, lucas_csv_path, 
                                    climate_csv_folder_path, l8_bands=bands, 
                                    transform=test_transform, return_point_id=True)

    SEQ_LEN = test_ds_w_id[0][0][1].shape[0]

    CSV_FILES = [f for f in os.listdir(climate_csv_folder_path) if f.endswith('.csv')]
    NUM_CLIMATE_FEATURES = len(CSV_FILES)

    now = datetime.now()
    run_name = now.strftime("D_%Y_%m_%d_T_%H_%M")
    print("Current Date and Time:", run_name)
    # 创建数据加载器

    if not JUST_LSTM and not JUST_CORR:
        if USE_LSTM_BRANCH and not USE_SPECTRAL:
            model = SoilNetLSTM(use_glam=USE_SPATIAL_ATTENTION, cnn_arch=CNN_ARCHITECTURE, reg_version=REG_VERSION,
                                cnn_in_channels=len(bands), regresor_input_from_cnn=1024,#vim模型是128，其余是1024
                                lstm_n_features=NUM_CLIMATE_FEATURES, lstm_n_layers=2, lstm_out=128,
                                hidden_size=128, rnn_arch=RNN_ARCHITECTURE, seq_len=SEQ_LEN).to(device)
        elif USE_SPECTRAL:
                    model = SoilNetCORR(
                    use_glam=USE_SPATIAL_ATTENTION,
                    cnn_arch=CNN_ARCHITECTURE,
                    reg_version=REG_VERSION,
                    # spectral_weight=SPECTRAL_WEIGHT,
                    # image_weight=IMAGE_WEIGHT,
                    cnn_in_channels=len(bands),
                    regresor_input_from_cnn=1024,
                    lstm_n_features=NUM_CLIMATE_FEATURES,
                    lstm_n_layers=2,
                    lstm_out=128,
                    spectral_in_features=2000,  # 新增光谱数据输入维度
                    spectral_hidden=512,        # 新增光谱数据隐藏层维度
                    hidden_size=128,
                    rnn_arch=RNN_ARCHITECTURE,
                    seq_len=SEQ_LEN,
                    img_size=64,               # 默认图像大小
                    spectral_net_type= SPECTRAL_NET_TYPE  # 默认使用VGG作为光谱网络
                ).to(device)
        else:
            model = SoilNet(use_glam=USE_SPATIAL_ATTENTION, cnn_arch=CNN_ARCHITECTURE, reg_version=REG_VERSION,
                            cnn_in_channels=len(bands), regresor_input_from_cnn=1024, hidden_size=128).to(device)
    elif JUST_LSTM:
        model = SoilNetJustLSTM(use_glam=USE_SPATIAL_ATTENTION, cnn_arch=CNN_ARCHITECTURE, reg_version=REG_VERSION,
                                cnn_in_channels=len(bands), regresor_input_from_cnn=1024,
                                lstm_n_features=NUM_CLIMATE_FEATURES, lstm_n_layers=2, lstm_out=128,
                                hidden_size=128, rnn_arch=RNN_ARCHITECTURE, seq_len=SEQ_LEN).to(device)
    else:
        model = SoilNetJustCORR(
           use_glam=USE_SPATIAL_ATTENTION,
            cnn_arch=CNN_ARCHITECTURE,
            reg_version=REG_VERSION,
            cnn_in_channels=len(bands),
            regresor_input_from_cnn=1024,
            lstm_n_features=NUM_CLIMATE_FEATURES,
            lstm_n_layers=2,
            lstm_out=128,
            spectral_in_features=SPECTRAL_IN_FEATURES,
            spectral_hidden=SPECTRAL_HIDDEN,
            hidden_size=128,
            rnn_arch=RNN_ARCHITECTURE,
            seq_len=SEQ_LEN,
            img_size=64,
            spectral_net_type=SPECTRAL_NET_TYPE
        ).to(device)
           
    if LOAD_SIMCLR_MODEL:
        model = torch.load(SIMCLR_PATH).to(device)
        if USE_SPECTRAL:
            model=SoilNetSimCLRwRegHead_CORR(model,hidden_size=128,reg_version=REG_VERSION).to(device)
        else:
            model = SoilNetSimCLRwRegHead(model, hidden_size=128, reg_version=REG_VERSION).to(device)


    # Load the best model
    BEST_MODEL_PATH = "result/RUN_LUCAS_Transformer_NoImage_D_2025_07_08_T_20_10_worst.pth.tar"
    load_checkpoint(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), filename=BEST_MODEL_PATH)
    model.eval()
    print("Best Model loaded")
    # 在调用模型之前，打印输入数据的结构

    # 创建数据加载器
    test_dl_w_id = DataLoader(test_ds_w_id, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,collate_fn=custom_collate_fn)
    # test_dl_w_id = DataLoader(test_ds_w_id, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    # 调用 test_step_w_id
    test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=True, csv_file=f"result/RUN_{EXP_NAME}_{run_name}_best.csv")
    print(f"Best model saved to result/RUN_{EXP_NAME}_{run_name}_best.csv")

    # Load the worst model
    WORST_MODEL_PATH = "result/RUN_LUCAS_Transformer_NoImage_D_2025_07_08_T_20_10_worst.pth.tar"
    load_checkpoint(model=model, optimizer=torch.optim.Adam(model.parameters(), lr=LEARNING_RATE), filename=WORST_MODEL_PATH)
    model.eval()
    print("Worst Model loaded")

    # 调用 test_step_w_id
    test_step_w_id(model=model, data_loader=test_dl_w_id, loss_fn=nn.L1Loss(), verbose=True, csv_file=f"result/RUN_{EXP_NAME}_{run_name}_worst.csv")
    print(f"Worst Model saved to result/RUN_{EXP_NAME}_{run_name}_worst.csv")

    # 计算最佳模型的评价指标
    df = pd.read_csv(f"result/RUN_{EXP_NAME}_{run_name}_best.csv")
    y_true = df['y_real'].values * OC_MAX
    y_pred = df['y_pred'].values * OC_MAX

    rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(y_true, y_pred)

    best_dict = {
        'RMSE': rmse,
        'R2': r2,
        'RPIQ': rpiq,
        'MAE': mae,
        'MEC': mec,
        'CCC': ccc
    }

    # 计算最差模型的评价指标
    df = pd.read_csv(f"result/RUN_{EXP_NAME}_{run_name}_worst.csv")
    y_true = df['y_real'].values * OC_MAX
    y_pred = df['y_pred'].values * OC_MAX

    rmse, r2, rpiq, mae, mec, ccc = evaluate_regression_metrics(y_true, y_pred)

    worst_dict = {
        'RMSE': rmse,
        'R2': r2,
        'RPIQ': rpiq,
        'MAE': mae,
        'MEC': mec,
        'CCC': ccc
    }

    # 保存结果到 JSON 文件
    cv_results_full = {
        'best_dict': best_dict,
        'worst_dict': worst_dict
    }

    with open(f"result/RUN_{EXP_NAME}_{run_name}.json", "w") as fp:
        json.dump(cv_results_full, fp, indent=4)
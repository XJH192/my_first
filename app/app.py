
from torchvision import transforms
import os
import sys
import io
import json
import zipfile
import shutil
import tempfile
import re
from datetime import datetime
from pathlib import Path

# 获取项目根目录（兼容 app.py 位于 app/ 子目录）
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(PROJECT_ROOT, 'soilnet')):
    PROJECT_ROOT = os.path.dirname(PROJECT_ROOT)

# 固定临时目录到项目内可写路径，避免系统 TEMP 权限问题
RUNTIME_TMP_DIR = os.path.join(PROJECT_ROOT, 'tmp_runtime')
RUNTIME_MPL_DIR = os.path.join(RUNTIME_TMP_DIR, 'mpl')
os.makedirs(RUNTIME_MPL_DIR, exist_ok=True)
os.environ['TMP'] = RUNTIME_TMP_DIR
os.environ['TEMP'] = RUNTIME_TMP_DIR
os.environ['TMPDIR'] = RUNTIME_TMP_DIR
os.environ['MPLCONFIGDIR'] = RUNTIME_MPL_DIR

# 添加项目根目录和 dataset 目录到 Python 路径
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'dataset'))

import base64
import io

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

# 兼容老版本 torch（如 1.8）缺少 torch.amp 的情况
if "torch.amp" not in sys.modules:
    try:
        from torch.amp import autocast as _autocast
    except Exception:
        import types

        def _autocast(device_type="cuda", dtype=None, enabled=True, cache_enabled=None):
            # 兼容 torch<1.10：忽略 device_type/dtype 等新参数
            try:
                from torch.cuda.amp import autocast as _cuda_autocast
                return _cuda_autocast(enabled=enabled)
            except Exception:
                from contextlib import contextmanager

                @contextmanager
                def _null_ctx():
                    yield

                return _null_ctx()

        _torch_amp = types.ModuleType("torch.amp")
        _torch_amp.autocast = _autocast
        sys.modules["torch.amp"] = _torch_amp
        # 兼容 `from torch import amp` 这类导入方式
        setattr(torch, "amp", _torch_amp)

import config
from soilnet.soil_net import SoilNetCORR
from dataset.dataset_loader import SNDatasetCorr, myNormalize,myToTensor

app = Flask(__name__)
CORS(app)

# ============ 配置 ============
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = r"D:\Soilnet(1)\result\ViT+trans+lstmcnn.tar"
OC_MAX = 560.2

# 气候变量配置：(key, 中文名, scale缩放系数, 单位, 保留小数)
# scale：原始整数值 × scale = 真实物理量（来自 TerraClimate / GEE 官方文档）
CLIMATE_VARS = [
    ('pr',   '降水量',       1.0,   'mm',   1),
    ('tmmx', '最高温度',     0.1,   '°C',   1),
    ('tmmn', '最低温度',     0.1,   '°C',   1),
    ('srad', '太阳辐射',     0.1,   'W/m²', 1),
    ('vpd',  '水汽压亏缺',   0.01,  'kPa',  3),
    ('vap',  '水汽压',       0.001, 'kPa',  3),
    ('pet',  '潜在蒸散发',   0.1,   'mm',   1),
    ('aet',  '实际蒸散发',   0.1,   'mm',   1),
    ('def',  '水分亏缺',     0.1,   'mm',   1),
    ('pdsi', 'Palmer干旱指数',0.01, '',     2),
    ('swe',  '雪水当量',     1.0,   'mm',   1),
    ('ro',   '径流',         1.0,   'mm',   1),
    ('soil', '土壤水分',     0.1,   'mm',   1),
]

# 模型配置（与 test.py 一致）
USE_SRTM = True
USE_LSTM_BRANCH = True
USE_SPECTRAL = True
SPECTRAL_NET_TYPE = 'lstmcnn'
CNN_ARCHITECTURE = "ViT"
RNN_ARCHITECTURE = 'Transformer'
REG_VERSION = 1
NUM_CLIMATE_FEATURES = 13
SEQ_LEN = 61

# 全局模型和数据集
current_model = None
current_dataset = None


def load_model():
    """加载预训练模型"""
    global current_model
    
    bands = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] if USE_SRTM else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    
    current_model = SoilNetCORR(
        use_glam=False,
        cnn_arch=CNN_ARCHITECTURE,
        reg_version=REG_VERSION,
        cnn_in_channels=len(bands),
        regresor_input_from_cnn=1024,
        lstm_n_features=NUM_CLIMATE_FEATURES,
        lstm_n_layers=2,
        lstm_out=128,
        spectral_in_features=2000,
        spectral_hidden=512,
        hidden_size=128,
        rnn_arch=RNN_ARCHITECTURE,
        seq_len=SEQ_LEN,
        img_size=64,
        spectral_net_type=SPECTRAL_NET_TYPE
    ).to(DEVICE)
    
    # 加载权重
    if os.path.exists(MODEL_PATH):
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        if 'state_dict' in checkpoint:
            current_model.load_state_dict(checkpoint['state_dict'])
        else:
            current_model.load_state_dict(checkpoint)
        current_model.eval()
        print(f"模型加载成功: {MODEL_PATH}")
    else:
        print(f"警告: 模型文件不存在 {MODEL_PATH}")
    
    return current_model


class NormalizeAndResize:
    """组合归一化和resize的transform"""
    def __init__(self, img_bands_min_max, oc_min, oc_max, output_size=(64, 64)):
        self.normalize = myNormalize(
            img_bands_min_max=img_bands_min_max,
            oc_min=oc_min,
            oc_max=oc_max
        )
        self.output_size = output_size
    
    def __call__(self, sample):
        img, oc = self.normalize(sample)
        # Resize 图像到 64x64
        if isinstance(img, torch.Tensor):
            # img shape: (bands, H, W)
            # 使用 interpolate 进行 resize
            img = torch.nn.functional.interpolate(
                img.unsqueeze(0),  # (1, bands, H, W)
                size=self.output_size,
                mode='bilinear',
                align_corners=False
            ).squeeze(0)  # (bands, 64, 64)
        return img, oc


def create_dataset(l8_dir, climate_dir, spectral_path, labels_path=None):
    # 定义完整的 transform 流程
    transform = transforms.Compose([
        myNormalize(
            img_bands_min_max=[[(0,7),(0,1)], [(7,12),(-1,1)], [(12), (-4,2963)], [(13), (0, 90)]],
            oc_min=0,
            oc_max=OC_MAX
        ),
        myToTensor()   # 关键：转为 tensor 并 resize 到 64x64
    ])
    
    # 如果 labels_path 不存在，创建一个临时的
    if labels_path is None or not os.path.exists(labels_path):
        # 从图像文件名中提取 point_id，创建临时标签文件
        l8_names = [f for f in os.listdir(l8_dir) if f.endswith('.tif')]
        if l8_names:
            point_ids = []
            for name in l8_names:
                pid = name.split('.')[0]
                try:
                    point_ids.append(int(pid))
                except:
                    continue
            
            # 创建临时标签 DataFrame
            labels_df = pd.DataFrame({
                'Point_ID': point_ids,
                'OC': [0.0] * len(point_ids)  # 预测时不需要真实OC
            })
            
            # 保存到临时文件
            temp_labels = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
            labels_df.to_csv(temp_labels.name, index=False)
            labels_path = temp_labels.name
    
    dataset = SNDatasetCorr(
        l8_dir=l8_dir,
        csv_dir=labels_path,
        climate_csv_folder=climate_dir,
        spectral_csv_path=spectral_path,
        l8_bands=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13] if USE_SRTM else [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        transform=transform,
        normalize_climate=True,  # 使用全局归一化
        normalize_spectral=True,  # 使用 StandardScaler
        return_point_id=True
    )
    
    return dataset


def predict_from_dataset(dataset, point_id):
    """
    使用 SNDatasetCorr 预测单个采样点
    """
    global current_model
    
    if current_model is None:
        load_model()
    
    # 找到对应的索引
    idx = None
    for i in range(len(dataset)):
        _, _, pid = dataset[i]
        if str(pid) == str(point_id) or str(pid) == str(int(float(point_id))):
            idx = i
            break
    
    if idx is None:
        raise ValueError(f"Point_ID {point_id} 在数据集中未找到")
    
    # 获取数据
    (image, climate, spectral), oc, pid = dataset[idx]
    
    # 确保数据是 tensor
    if not isinstance(image, torch.Tensor):
        image = torch.from_numpy(image).float()
    if not isinstance(climate, torch.Tensor):
        climate = torch.from_numpy(climate).float()
    if not isinstance(spectral, torch.Tensor):
        spectral = torch.from_numpy(spectral).float()
    
    # 检查数据是否含 NaN
    if torch.isnan(image).any():
        raise ValueError(f"图像数据包含NaN")
    if torch.isnan(climate).any():
        raise ValueError(f"气候数据包含NaN")
    if torch.isnan(spectral).any():
        raise ValueError(f"光谱数据包含NaN")
    
    # 预测
    with torch.no_grad():
        image = image.unsqueeze(0).to(DEVICE)
        climate = climate.unsqueeze(0).to(DEVICE)
        spectral = spectral.unsqueeze(0).to(DEVICE)
        
        y_pred = current_model([image, climate, spectral])
        
        if torch.isnan(y_pred).any():
            raise ValueError("模型预测结果为NaN")
        
        soc_pred = y_pred.item() * OC_MAX
    
    return soc_pred, oc, climate.numpy(), spectral.numpy()


# ============ 路由 ============

@app.route('/')
def index():
    """首页重定向到 dashboard2"""
    return send_from_directory('.', 'dashboard.html')


@app.route('/dashboard.html')
def serve_dashboard():
    """大屏页面"""
    return send_from_directory('.', 'dashboard.html')


@app.route('/api/status')
def api_status():
    """检查模型状态"""
    global current_model
    if current_model is None:
        load_model()
    
    return jsonify({
        'model_loaded': current_model is not None,
        'device': DEVICE,
        'model_path': MODEL_PATH
    })


@app.route('/api/image/preview', methods=['POST'])
def api_image_preview():
    """读取 .tif 文件，合成 RGB 伪彩色图，返回 base64 PNG"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': '缺少图像文件'}), 400

        image_file = request.files['image']
        filename   = image_file.filename
        point_id   = filename.split('.')[0]

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp:
            image_file.save(tmp.name)
            tmp_path = tmp.name

        try:
            with rasterio.open(tmp_path) as ds:
                data = ds.read()  # (bands, H, W)

            n_bands = data.shape[0]
            # Landsat8: Band4=红(idx3), Band3=绿(idx2), Band2=蓝(idx1)
            # 若波段不足则降级使用前三波段
            r_idx = min(3, n_bands - 1)
            g_idx = min(2, n_bands - 1)
            b_idx = min(1, n_bands - 1)

            def stretch(band):
                p2, p98 = np.percentile(band[~np.isnan(band)] if np.isnan(band).any() else band, [2, 98])
                clipped = np.clip(band, p2, p98)
                rng = p98 - p2
                if rng < 1e-9:
                    return np.zeros_like(band, dtype=np.uint8)
                return ((clipped - p2) / rng * 255).astype(np.uint8)

            rgb = np.stack([stretch(data[r_idx].astype(float)),
                            stretch(data[g_idx].astype(float)),
                            stretch(data[b_idx].astype(float))], axis=-1)

            img = Image.fromarray(rgb, mode='RGB')
            img = img.resize((240, 240), Image.LANCZOS)

            buf = io.BytesIO()
            img.save(buf, format='PNG')
            b64 = 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()

            return jsonify({'success': True, 'image_base64': b64, 'point_id': point_id,
                            'shape': list(data.shape)})
        finally:
            os.unlink(tmp_path)

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500



@app.route('/api/predict', methods=['POST'])
def api_predict():
    """单点预测 API"""
    try:
        # 检查文件
        if 'image' not in request.files:
            return jsonify({'error': '缺少图像文件'}), 400
        
        image_file = request.files['image']
        
        # 提取 point_id
        filename = image_file.filename
        point_id = filename.split('.')[0]
        
        # 创建临时目录
        with tempfile.TemporaryDirectory() as temp_dir:
            # 保存图像
            img_dir = os.path.join(temp_dir, 'images')
            os.makedirs(img_dir, exist_ok=True)
            img_path = os.path.join(img_dir, filename)
            image_file.save(img_path)
            
            # 处理气候数据
            climate_dir = None
            if 'climate' in request.files:
                climate_zip = request.files['climate']
                climate_dir = os.path.join(temp_dir, 'climate')
                os.makedirs(climate_dir, exist_ok=True)
                
                with zipfile.ZipFile(climate_zip, 'r') as z:
                    z.extractall(climate_dir)
                
                # 找到包含 LUCAS_*.csv 的目录
                for root, dirs, files in os.walk(climate_dir):
                    if any(f.endswith('.csv') for f in files):
                        climate_dir = root
                        break
            
            # 处理光谱数据
            spectral_path = None
            if 'spectral' in request.files:
                spectral_file = request.files['spectral']
                spectral_path = os.path.join(temp_dir, 'spectral.csv')
                spectral_file.save(spectral_path)
            
            # 检查必需文件
            if climate_dir is None or not os.path.exists(climate_dir):
                return jsonify({'error': '缺少气候数据'}), 400
            if spectral_path is None or not os.path.exists(spectral_path):
                return jsonify({'error': '缺少光谱数据'}), 400
            
            # 创建数据集并预测
            dataset = create_dataset(img_dir, climate_dir, spectral_path)
            soc_pred, oc_true, climate_data, spectral_data = predict_from_dataset(dataset, point_id)
            
            return jsonify({
                'success': True,
                'point_id': point_id,
                'soc_prediction': round(soc_pred, 2),
                'oc_true': round(oc_true, 2) if oc_true > 0 else None,
                'climate_shape': climate_data.shape,
                'spectral_shape': spectral_data.shape
            })
    
    except Exception as e:
        import traceback
        print(f"预测错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


@app.route('/api/climate/parse', methods=['POST'])
def api_parse_climate():
    """解析气候数据用于可视化"""
    try:
        if 'climate_zip' not in request.files:
            return jsonify({'error': '缺少气候ZIP文件'}), 400
        
        climate_zip = request.files['climate_zip']
        point_id = request.form.get('point_id', '')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            climate_dir = os.path.join(temp_dir, 'climate')
            os.makedirs(climate_dir, exist_ok=True)
            
            with zipfile.ZipFile(climate_zip, 'r') as z:
                z.extractall(climate_dir)
            
            # 找到包含 LUCAS_*.csv 的目录
            for root, dirs, files in os.walk(climate_dir):
                if any(f.endswith('.csv') for f in files):
                    climate_dir = root
                    break
            
            # 读取气候数据，应用 TerraClimate 缩放系数，返回真实物理量
            climate_data = {}
            pid_float = float(point_id)
            pid_int = int(pid_float)
            
            for var_key, var_name, scale, unit, decimals in CLIMATE_VARS:
                csv_path = os.path.join(climate_dir, f"LUCAS_{var_key}_processed.csv")
                if not os.path.exists(csv_path):
                    continue
                
                df = pd.read_csv(csv_path)
                time_cols = [col for col in df.columns if str(col).startswith('20') and str(col).isdigit() and len(str(col)) == 8]
                
                row = df[df['Point_ID'] == pid_float]
                if row.empty:
                    row = df[df['Point_ID'] == pid_int]
                
                if row.empty:
                    values = [0.0] * 61
                else:
                    raw = row[time_cols].values[0].astype(float)
                    values = [round(v * scale, decimals) for v in raw]
                
                climate_data[var_key] = {
                    'name': var_name,
                    'unit': unit,
                    'values': values,
                    'dates': time_cols
                }
            
            return jsonify({'success': True, 'climate': climate_data})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/spectral/parse', methods=['POST'])
def api_parse_spectral():
    """解析光谱数据用于可视化"""
    try:
        if 'spectral' not in request.files:
            return jsonify({'error': '缺少光谱文件'}), 400
        
        spectral_file = request.files['spectral']
        point_id = request.form.get('point_id', '')
        
        with tempfile.TemporaryDirectory() as temp_dir:
            spectral_path = os.path.join(temp_dir, 'spectral.csv')
            spectral_file.save(spectral_path)
            
            df = pd.read_csv(spectral_path)
            wavelength_cols = [col for col in df.columns if str(col).isdigit()]
            wavelengths = [int(w) for w in wavelength_cols]
            
            id_col = None
            for c in ['PointID', 'Point_ID']:
                if c in df.columns:
                    id_col = c
                    break
            
            pid_float = float(point_id)
            pid_int = int(pid_float)
            
            row = df[df[id_col] == pid_float]
            if row.empty:
                row = df[df[id_col] == pid_int]
            
            if row.empty:
                return jsonify({'error': f'PointID {point_id} 未找到'}), 404
            
            values = row[wavelength_cols].values[0].astype(float).tolist()
            
            return jsonify({
                'success': True,
                'wavelengths': wavelengths,
                'values': values
            })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/predict_batch', methods=['POST'])
def api_predict_batch():
    """批量预测 API（与单点预测保持同源数据处理流程）"""
    try:
        if 'batch_zip' not in request.files:
            return jsonify({'error': '缺少批量数据ZIP文件'}), 400
        
        global current_model
        if current_model is None:
            load_model()

        batch_zip = request.files['batch_zip']
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # 1) 解压全部文件
            with zipfile.ZipFile(batch_zip, 'r') as z:
                z.extractall(temp_dir)

            # 2) 扫描文件并分流
            tif_files = []
            csv_files = []
            for root, _, files in os.walk(temp_dir):
                for f in files:
                    f_lower = f.lower()
                    full_path = os.path.join(root, f)
                    if f_lower.endswith('.tif'):
                        tif_files.append(full_path)
                    elif f_lower.endswith('.csv'):
                        csv_files.append(full_path)

            if not tif_files:
                return jsonify({'error': 'ZIP中缺少图像文件（.tif）'}), 400

            # 3) 准备 images 目录（与单点保持一致输入形式）
            img_dir = os.path.join(temp_dir, '_batch_images')
            os.makedirs(img_dir, exist_ok=True)
            seen_img_names = set()
            for src in tif_files:
                name = os.path.basename(src)
                if name in seen_img_names:
                    continue
                shutil.copy2(src, os.path.join(img_dir, name))
                seen_img_names.add(name)

            # 4) 严格筛选 13 个气候文件，避免把光谱文件误当气候文件
            expected_climate = {
                f"lucas_{k}_processed.csv": f"LUCAS_{k}_processed.csv"
                for k, _, _, _, _ in CLIMATE_VARS
            }
            climate_sources = {}
            spectral_candidates = []
            labels_path = None

            for csv_path in csv_files:
                fname = os.path.basename(csv_path)
                fname_lower = fname.lower()

                if fname_lower in expected_climate:
                    climate_sources[fname_lower] = csv_path
                elif 'label' in fname_lower and labels_path is None:
                    labels_path = csv_path
                else:
                    spectral_candidates.append(csv_path)

            missing_climate = [
                expected_climate[k] for k in expected_climate.keys() if k not in climate_sources
            ]
            if missing_climate:
                return jsonify({
                    'error': f"气候数据不完整，缺少文件: {', '.join(missing_climate)}"
                }), 400

            climate_dir = os.path.join(temp_dir, '_batch_climate')
            os.makedirs(climate_dir, exist_ok=True)
            for k_lower, canonical_name in expected_climate.items():
                shutil.copy2(climate_sources[k_lower], os.path.join(climate_dir, canonical_name))

            # 5) 选择光谱文件（优先常见文件名）
            if not spectral_candidates:
                return jsonify({'error': 'ZIP中缺少光谱CSV文件'}), 400

            def _spectral_rank(path):
                name = os.path.basename(path).lower()
                if name in ('lucas_2015_all.csv', 'lucas2015corr_use.csv'):
                    return 0
                if 'spectral' in name:
                    return 1
                if '2015_all' in name or 'corr_use' in name:
                    return 2
                return 3

            spectral_candidates.sort(key=_spectral_rank)
            spectral_path = spectral_candidates[0]

            # 6) 预读取批量气候时序（用于前端下拉查看各采样点）
            climate_meta = {}
            for var_key, var_name, scale, unit, decimals in CLIMATE_VARS:
                csv_path = os.path.join(climate_dir, f"LUCAS_{var_key}_processed.csv")
                df = pd.read_csv(csv_path)
                date_cols = [
                    col for col in df.columns
                    if str(col).startswith('20') and str(col).isdigit() and len(str(col)) == 8
                ]
                climate_meta[var_key] = {
                    'df': df,
                    'date_cols': date_cols,
                    'dates': [str(c) for c in date_cols],
                    'name': var_name,
                    'unit': unit,
                    'scale': float(scale),
                    'decimals': int(decimals)
                }

            def _match_climate_row(df, pid_str):
                if 'Point_ID' not in df.columns:
                    return None

                pid_series = df['Point_ID']
                pid_raw = str(pid_str).strip()

                # 生成候选 Point_ID（兼容 26601784.0_150101、26601784.0、26601784 等形式）
                candidates = []
                candidates.append(pid_raw)
                if '_' in pid_raw:
                    candidates.append(pid_raw.split('_')[0])

                m = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)', pid_raw)
                if m:
                    candidates.append(m.group(1))

                # 去重并保序
                dedup = []
                seen = set()
                for c in candidates:
                    c = str(c).strip()
                    if c and c not in seen:
                        dedup.append(c)
                        seen.add(c)

                row = df.iloc[0:0]

                for c in dedup:
                    # 数值匹配
                    try:
                        c_float = float(c)
                        c_int = int(c_float)
                        row = df[pid_series == c_float]
                        if not row.empty:
                            break
                        row = df[pid_series == c_int]
                        if not row.empty:
                            break
                    except Exception:
                        pass

                    # 字符串匹配
                    row = df[pid_series.astype(str).str.strip() == c]
                    if not row.empty:
                        break

                if row.empty:
                    return None
                return row.iloc[0]

            def _build_point_climate(pid_str):
                point_climate = {}
                for var_key, meta in climate_meta.items():
                    row = _match_climate_row(meta['df'], pid_str)
                    if row is None:
                        # 缺失时返回空值，避免误显示为 0
                        values = [None] * len(meta['date_cols'])
                    else:
                        raw_series = pd.to_numeric(row[meta['date_cols']], errors='coerce')
                        values = []
                        for v in raw_series.tolist():
                            if pd.isna(v):
                                values.append(None)
                            else:
                                values.append(round(float(v) * meta['scale'], meta['decimals']))

                    point_climate[var_key] = {
                        'name': meta['name'],
                        'unit': meta['unit'],
                        'dates': meta['dates'],
                        'values': values
                    }
                return point_climate

            # 7) 创建数据集并批量推理（复用单点同样模型输入流程）
            dataset = create_dataset(img_dir, climate_dir, spectral_path, labels_path)
            results = []
            batch_climate = {}
            for i in range(len(dataset)):
                point_id = 'unknown'
                try:
                    (image, climate, spectral), oc, point_id = dataset[i]

                    if not isinstance(image, torch.Tensor):
                        image = torch.from_numpy(image).float()
                    if not isinstance(climate, torch.Tensor):
                        climate = torch.from_numpy(climate).float()
                    if not isinstance(spectral, torch.Tensor):
                        spectral = torch.from_numpy(spectral).float()

                    if torch.isnan(image).any():
                        raise ValueError("图像数据包含NaN")
                    if torch.isnan(climate).any():
                        raise ValueError("气候数据包含NaN")
                    if torch.isnan(spectral).any():
                        raise ValueError("光谱数据包含NaN")

                    with torch.no_grad():
                        image = image.unsqueeze(0).to(DEVICE)
                        climate = climate.unsqueeze(0).to(DEVICE)
                        spectral = spectral.unsqueeze(0).to(DEVICE)

                        y_pred = current_model([image, climate, spectral])
                        if torch.isnan(y_pred).any():
                            raise ValueError("模型预测结果为NaN")
                        soc_pred = y_pred.item() * OC_MAX

                    point_id_str = str(point_id)
                    if point_id_str not in batch_climate:
                        batch_climate[point_id_str] = _build_point_climate(point_id_str)

                    results.append({
                        'point_id': point_id_str,
                        'soc_prediction': round(soc_pred, 2),
                        'oc_true': round(float(oc), 2) if float(oc) > 0 else None
                    })
                except Exception as e:
                    results.append({
                        'point_id': str(point_id),
                        'error': str(e)
                    })
            
            return jsonify({
                'success': True,
                'count': len(results),
                'results': results,
                'batch_climate': batch_climate
            })
    
    except Exception as e:
        import traceback
        print(f"批量预测错误: {str(e)}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    # 启动时加载模型
    load_model()
    app.run(host='0.0.0.0', port=5000, debug=True)

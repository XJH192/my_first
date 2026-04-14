"""
extract_sample.py - 从 LUCAS 数据集中提取单个采样点的三路数据，生成可上传大屏的文件

用法：
    python extract_sample.py --point_id 32421754
    python extract_sample.py --point_id 32421754 --out_dir my_data/

输出：
    {out_dir}/climate_{point_id}.csv   - 13行×61列气候数据（可直接上传大屏）
    {out_dir}/spectral_{point_id}.csv  - 1行×2107列光谱数据（可直接上传大屏）
    {out_dir}/image_{point_id}.tif     - 遥感图像（如果存在于测试集中则复制过来）

注意：遥感图像需要你自己准备，脚本只能复制测试集已有的图像。
"""

import os
import sys
import argparse
import shutil
import io

import pandas as pd
import numpy as np

# ── 项目根目录 ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── 数据路径 ──
CLIMATE_DIR  = os.path.join(BASE_DIR, "dataset", "Climate", "All")
SPECTRAL_CSV = os.path.join(BASE_DIR, "dataset", "lucas2015corr_use.csv")
LUCAS_CSV    = os.path.join(BASE_DIR, "dataset", "LUCAS_2015_all.csv")
IMG_DIR      = os.path.join(BASE_DIR, "dataset", "l8_images", "test")

# ── 13个气候变量文件（顺序固定）──
VAR_FILES = [
    ("tmmx",  "LUCAS_tmmx_processed.csv"),
    ("tmmn",  "LUCAS_tmmn_processed.csv"),
    ("pr",    "LUCAS_pr_processed.csv"),
    ("pet",   "LUCAS_pet_processed.csv"),
    ("aet",   "LUCAS_aet_processed.csv"),
    ("vpd",   "LUCAS_vpd_processed.csv"),
    ("vap",   "LUCAS_vap_processed.csv"),
    ("srad",  "LUCAS_srad_processed.csv"),
    ("swe",   "LUCAS_swe_processed.csv"),
    ("soil",  "LUCAS_soil_processed.csv"),
    ("ro",    "LUCAS_ro_processed.csv"),
    ("def",   "LUCAS_def_processed.csv"),
    ("pdsi",  "LUCAS_pdsi_processed.csv"),
]


def list_available_points(n=20):
    """列出数据集中可用的采样点"""
    df = pd.read_csv(LUCAS_CSV, nrows=n+1)
    id_col = "Point_ID" if "Point_ID" in df.columns else df.columns[0]
    print(f"\n数据集中前{n}个采样点 (Point_ID)：")
    for pid in df[id_col].head(n).tolist():
        print(f"  {pid}")
    print(f"\n共{len(df)}行（只显示前{n}个）")


def extract_climate(point_id: int, out_path: str):
    """从13个气候文件中提取指定 Point_ID 的数据，保存为 13行×61列 CSV"""
    rows = []
    var_names = []
    date_cols_ref = None

    for var_name, fname in VAR_FILES:
        fpath = os.path.join(CLIMATE_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  [警告] 找不到文件：{fpath}，跳过")
            continue
        df = pd.read_csv(fpath)
        # 找日期列（8位数字字符串）
        dcols = [c for c in df.columns if str(c).isdigit() and len(str(c)) == 8]
        if date_cols_ref is None:
            date_cols_ref = dcols

        # 按 Point_ID 查找
        if "Point_ID" in df.columns:
            mask = df["Point_ID"] == point_id
            if mask.sum() == 0:
                print(f"  [警告] {fname} 中找不到 Point_ID={point_id}，用第1行代替")
                row = df[dcols].iloc[0].values.astype(float)
            else:
                row = df.loc[mask, dcols].iloc[0].values.astype(float)
        else:
            print(f"  [警告] {fname} 没有 Point_ID 列，用第1行")
            row = df[dcols].iloc[0].values.astype(float)

        rows.append(row)
        var_names.append(var_name)

    # 构建 DataFrame：13行 × 61列，行索引 = 变量名
    out_df = pd.DataFrame(rows, index=var_names, columns=date_cols_ref)
    out_df.index.name = "variable"
    out_df.to_csv(out_path)
    print(f"  [OK] 气候数据已保存：{out_path}  ({len(rows)} 变量 × {len(date_cols_ref)} 时间步)")


def extract_spectral(point_id: int, out_path: str):
    """从 lucas2015corr_use.csv 中提取指定 Point_ID 的光谱数据"""
    if not os.path.exists(SPECTRAL_CSV):
        print(f"  [错误] 找不到光谱文件：{SPECTRAL_CSV}")
        return False

    df = pd.read_csv(SPECTRAL_CSV)
    # 光谱文件的ID列名是 PointID（无下划线），也兼容 Point_ID
    id_col = None
    for candidate in ("PointID", "Point_ID", "pointid", "point_id"):
        if candidate in df.columns:
            id_col = candidate
            break

    if id_col and point_id in df[id_col].values:
        mask = df[id_col] == point_id
        row = df[mask].iloc[0]
    else:
        if id_col:
            print(f"  [警告] 光谱文件中找不到 Point_ID={point_id}，用第1行代替")
        else:
            print(f"  [警告] 光谱文件中没有ID列，用第1行代替")
        row = df.iloc[0]

    # 只保留波长列（整数列名）
    wave_cols = [c for c in df.columns if str(c).isdigit()]
    out_df = row[wave_cols].to_frame().T
    out_df.to_csv(out_path, index=False)
    print(f"  [OK] 光谱数据已保存：{out_path}  ({len(wave_cols)} 波段)")
    return True


def find_image(point_id: int, out_path: str):
    """尝试在测试集中找到该 Point_ID 对应的遥感图像"""
    candidates = [
        os.path.join(IMG_DIR, f"{point_id}.tif"),
        os.path.join(IMG_DIR, f"{point_id}_l8.tif"),
    ]
    for c in candidates:
        if os.path.exists(c):
            shutil.copy2(c, out_path)
            print(f"  [OK] 遥感图像已复制：{out_path}")
            return True

    # 在 img_dir 里搜索（文件名包含 point_id）
    if os.path.exists(IMG_DIR):
        for fname in os.listdir(IMG_DIR):
            if str(point_id) in fname and fname.endswith(".tif"):
                shutil.copy2(os.path.join(IMG_DIR, fname), out_path)
                print(f"  [OK] 遥感图像已复制：{out_path}（来源：{fname}）")
                return True

    print(f"  [未找到] 测试集中没有 Point_ID={point_id} 的遥感图像")
    print(f"           请手动准备 14波段 Landsat 8 .tif 文件")
    return False


def main():
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="从LUCAS数据集提取单点三路数据")
    parser.add_argument("--point_id", type=int, default=None, help="采样点 Point_ID（整数）")
    parser.add_argument("--out_dir",  type=str, default="sample_data", help="输出目录")
    parser.add_argument("--list",     action="store_true", help="列出可用的采样点")
    args = parser.parse_args()

    if args.list:
        list_available_points(30)
        return

    if args.point_id is None:
        print("错误：请提供 --point_id 参数，或使用 --list 查看可用采样点")
        print("示例：python extract_sample.py --point_id 32421754")
        return

    pid = args.point_id
    out_dir = os.path.join(BASE_DIR, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n正在提取 Point_ID = {pid} 的数据...")
    print(f"输出目录：{out_dir}\n")

    clim_out = os.path.join(out_dir, f"climate_{pid}.csv")
    spec_out = os.path.join(out_dir, f"spectral_{pid}.csv")
    img_out  = os.path.join(out_dir, f"image_{pid}.tif")

    extract_climate(pid, clim_out)
    extract_spectral(pid, spec_out)
    find_image(pid, img_out)

    print(f"""
提取完成！可上传到大屏的文件：

  气候数据：{clim_out}
  光谱数据：{spec_out}
  遥感图像：{img_out}

使用方式：
  1. 启动后端：D:\\.conda\\yaogan\\python.exe app.py
  2. 打开大屏：http://127.0.0.1:5000/dashboard.html
  3. 分别上传以上三个文件
  4. 点击"开始预测 SOC"
""")


if __name__ == "__main__":
    main()

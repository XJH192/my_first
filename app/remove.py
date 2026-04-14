# import os
# import glob
# from collections import defaultdict

# # 定义文件夹路径
# folders = ['train', 'val', 'test']  # 优先级顺序：train > val > test
# base_path = './dataset/l8_images/'  # 修改为您的实际路径

# # 查找所有tif文件及其位置
# file_locations = defaultdict(list)
# for folder in folders:
#     folder_path = os.path.join(base_path, folder)
#     if os.path.exists(folder_path):
#         for file in glob.glob(os.path.join(folder_path, "*.tif")):
#             filename = os.path.basename(file)
#             file_locations[filename].append(folder)

# # 打印重复文件信息
# print("\n检测到的重复文件:")
# duplicates = {filename: locations for filename, locations in file_locations.items() if len(locations) > 1}
# for filename, locations in duplicates.items():
#     print(f"{filename}: 出现在 {', '.join(locations)}")

# # 定义优先级：train > val > test
# priority = {folder: idx for idx, folder in enumerate(reversed(folders))}

# # 处理重复文件
# for filename, locations in duplicates.items():
#     # 按优先级排序位置
#     locations.sort(key=lambda x: priority.get(x, 999), reverse=True)
#     keep_folder = locations[0]  # 保留优先级最高的文件夹中的文件
    
#     # 从其他位置删除文件
#     for folder in locations[1:]:
#         file_path = os.path.join(base_path, folder, filename)
#         if os.path.exists(file_path):
#             os.remove(file_path)
#             print(f"已从 {folder} 删除 {filename}（保留在 {keep_folder}）")

# # 显示处理后的文件结构
# print("\n处理后的文件结构:")
# for folder in folders:
#     folder_path = os.path.join(base_path, folder)
#     files = glob.glob(os.path.join(folder_path, "*.tif"))
#     print(f"{folder}文件夹: {len(files)}个文件")

# # 验证train和test之间是否还有重复
# train_files = set([os.path.basename(f) for f in glob.glob(os.path.join(base_path, 'train', "*.tif"))])
# test_files = set([os.path.basename(f) for f in glob.glob(os.path.join(base_path, 'test', "*.tif"))])

# print("\n验证train和test之间的重复情况:")
# common_files = train_files.intersection(test_files)
# if common_files:
#     print(f"警告：train和test之间仍有{len(common_files)}个重复文件：{common_files}")
# else:
#     print("成功：train和test之间没有重复文件")
# import os
# import random
# import shutil
# import glob

# # 定义文件夹路径
# val_folder = './dataset/l8_images/val/'  # 替换为您的val文件夹路径
# test_folder = './dataset/l8_images/test/'  # 替换为您的test文件夹路径

# # 获取当前test文件夹中的文件列表
# test_files = glob.glob(os.path.join(test_folder, "*.tif"))
# current_test_count = len(test_files)
# print(f"当前test文件夹中有 {current_test_count} 个tif文件")

# # 计算需要复制的文件数量
# target_count = 4347
# files_to_copy = target_count - current_test_count

# if files_to_copy <= 0:
#     print(f"test文件夹已经有 {current_test_count} 个文件，已达到或超过目标数量 {target_count}")
# else:
#     print(f"需要从val文件夹复制 {files_to_copy} 个文件到test文件夹")
    
#     # 获取val文件夹中的所有tif文件
#     val_files = glob.glob(os.path.join(val_folder, "*.tif"))
#     print(f"val文件夹中有 {len(val_files)} 个tif文件")
    
#     # 获取test文件夹中已有文件的文件名（不包含路径）
#     test_filenames = set(os.path.basename(f) for f in test_files)
    
#     # 筛选出val中有但test中没有的文件
#     available_files = [f for f in val_files if os.path.basename(f) not in test_filenames]
#     print(f"val文件夹中有 {len(available_files)} 个文件可供复制（排除了与test重复的文件）")
    
#     if len(available_files) < files_to_copy:
#         print(f"警告：val文件夹中只有 {len(available_files)} 个可用文件，少于需要复制的 {files_to_copy} 个")
#         files_to_copy = len(available_files)
    
#     # 随机选择文件进行复制
#     selected_files = random.sample(available_files, files_to_copy)
    
#     # 复制文件
#     for file in selected_files:
#         filename = os.path.basename(file)
#         destination = os.path.join(test_folder, filename)
#         shutil.copy2(file, destination)
#         print(f"已复制: {filename}")
    
#     # 验证复制后的文件数量
#     final_test_count = len(glob.glob(os.path.join(test_folder, "*.tif")))
#     print(f"\n复制完成。test文件夹现在有 {final_test_count} 个tif文件")
import os
import glob
import pandas as pd

# 设置train和test文件夹的路径
# 请根据实际情况修改这些路径
train_folder = "./dataset/l8_images/train/"
test_folder = "./dataset/l8_images/test/"

# 获取所有tif文件名
train_files = [os.path.basename(f) for f in glob.glob(os.path.join(train_folder, "*.tif"))]
test_files = [os.path.basename(f) for f in glob.glob(os.path.join(test_folder, "*.tif"))]

# 提取point_id (文件名中.之前的部分)
def extract_point_id(filename):
    return filename.split('.')[0]

train_point_ids = {extract_point_id(f): f for f in train_files}
test_point_ids = {extract_point_id(f): f for f in test_files}

# 找出重复的point_ids
duplicates = set(train_point_ids.keys()) & set(test_point_ids.keys())

# 创建一个DataFrame来存储重复的point_id
df_duplicates = pd.DataFrame(sorted(duplicates), columns=['point_id'])

# 显示表格
print("重复文件的point_id表格:")
print(df_duplicates.to_string(index=False))

# 如果需要保存到CSV文件
df_duplicates.to_csv('duplicate_point_ids.csv', index=False)
import os
import shutil
from pathlib import Path

# 设置基础目录路径
base_dir = "./dataset/l8_images/"  # 修改为你的数据目录路径

# 构建完整路径
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
train_delete_dir = os.path.join(base_dir, 'train_delete')
val_delete_dir = os.path.join(base_dir, 'val_delete')

# 创建目标文件夹（如果不存在）
os.makedirs(train_delete_dir, exist_ok=True)
os.makedirs(val_delete_dir, exist_ok=True)

# 获取每个文件夹中的文件名（仅.tif文件）
train_files = set([f for f in os.listdir(train_dir) if f.endswith('.tif')])
val_files = set([f for f in os.listdir(val_dir) if f.endswith('.tif')])
test_files = set([f for f in os.listdir(test_dir) if f.endswith('.tif')])

print(f"原始文件数量: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

# 找出重复的文件
train_duplicates = set()
for file in train_files:
    if file in val_files or file in test_files:
        train_duplicates.add(file)

val_duplicates = set()
for file in val_files:
    if file in train_files or file in test_files:
        val_duplicates.add(file)

print(f"检测到的重复文件: train_duplicates={len(train_duplicates)}, val_duplicates={len(val_duplicates)}")

# 移动重复文件
for file in train_duplicates:
    src_path = os.path.join(train_dir, file)
    dst_path = os.path.join(train_delete_dir, file)
    shutil.move(src_path, dst_path)
    print(f"移动: {src_path} -> {dst_path}")

for file in val_duplicates:
    src_path = os.path.join(val_dir, file)
    dst_path = os.path.join(val_delete_dir, file)
    shutil.move(src_path, dst_path)
    print(f"移动: {src_path} -> {dst_path}")

# 打印移动后的结果
print("\n移动后各文件夹的文件数量：")
print(f"train: {len(os.listdir(train_dir))}")
print(f"val: {len(os.listdir(val_dir))}")
print(f"test: {len(os.listdir(test_dir))}")
print(f"train_delete: {len(os.listdir(train_delete_dir))}")
print(f"val_delete: {len(os.listdir(val_delete_dir))}")
# import os
# import shutil
# import random

# # 文件夹路径 - 请替换为您的实际路径
# train_dir = './dataset/l8_images/train'
# val_dir = './dataset/l8_images/val'
# test_dir = './dataset/l8_images/test'

# # 获取test目录中的所有文件
# test_files = [f for f in os.listdir(test_dir) if os.path.isfile(os.path.join(test_dir, f))]
# random.shuffle(test_files)  # 随机打乱文件列表

# # 移动文件到train
# files_to_train = test_files[:225]
# for file in files_to_train:
#     shutil.move(os.path.join(test_dir, file), os.path.join(train_dir, file))
    
# # 移动文件到val
# files_to_val = test_files[225:451]
# for file in files_to_val:
#     shutil.move(os.path.join(test_dir, file), os.path.join(val_dir, file))

# print('文件移动完成！')




import pandas as pd
from ultralytics import YOLO
import os
from collections import Counter

# ======================
# 配置路径
# ======================
base_dir = "/content/drive/MyDrive/show//10_lishui/03_unclassified"
csv_dir = os.path.join(base_dir)      # CSV 存放文件夹
img_dir = os.path.join(base_dir)    # 图片文件夹的总目录
output_dir = os.path.join("/content/drive/MyDrive/04_Predict/10_lishui/output_3") # 输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载模型
model = YOLO("/content/drive/MyDrive/03_AblationExperiment/CurriculumLearning★/detect/train/weights/best.pt")

# ======================
# 决策函数
# ======================
def decide_parking_type(detected_types):
    if not detected_types:
        return None
    counts = Counter(detected_types)
    max_count = max(counts.values())
    max_types = [t for t, cnt in counts.items() if cnt == max_count]
    if len(max_types) == 1:
        return max_types[0]
    for t in detected_types:
        if t in max_types:
            return t

# ======================
# 处理一组图片
# ======================
import re

def extract_lat_lon_from_filename(filename):
    """
    从文件名中提取经纬度，例如：
    118.78783315731_32.102459698168_170_0.jpg
    返回 (118.78783315731, 32.102459698168)
    """
    base = os.path.splitext(filename)[0]
    parts = base.split("_")
    if len(parts) >= 2:
        try:
            lon = float(parts[0])
            lat = float(parts[1])
            return (lon, lat)
        except ValueError:
            return None
    return None

def process_images_for_csv(df, img_folder, side_name, model):
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg','.jpeg','.png'))]

    # 创建一个字典：key=(lon, lat), value=预测结果
    lon_lat_to_result = {}

    for img_file in image_files:
        lon_lat = extract_lat_lon_from_filename(img_file)
        if lon_lat is None:
            continue

        img_full = os.path.join(img_folder, img_file)
        results = model.predict(source=img_full, save=False, verbose=False)

        detected_types = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls.cpu().item())
                    cls_name = model.names[cls_id]
                    detected_types.append(cls_name)

        parking_type = decide_parking_type(detected_types)
        lon_lat_to_result[lon_lat] = parking_type if parking_type else "UNKNOWN"

    # 根据 CSV 中的经纬度匹配结果
    results_list = []
    for _, row in df.iterrows():
        lon = float(row.iloc[0])
        lat = float(row.iloc[1])
        key = (lon, lat)
        results_list.append(lon_lat_to_result.get(key, "UNKNOWN"))

    df[side_name] = results_list
    return df

# ======================
# 主程序：遍历所有 CSV
# ======================
encodings = ["utf-8", "utf-8-sig", "gbk", "ansi", "latin1"]

for file in os.listdir(csv_dir):
    if not file.endswith(".csv"):
        continue

    csv_path = os.path.join(csv_dir, file)
    for enc in encodings:
      try:
        df = pd.read_csv(csv_path, encoding=enc)
        print(f"读取成功：{file},编码：{enc}")
        break
      except UnicodeDecodeError:
        continue
    else:
      print(f"❌ 无法读取文件: {file}")

    # 提取编号和方向
    # 示例：96_point_up.csv -> id=96, direction=up
    name = os.path.splitext(file)[0]
    parts = name.split("_")   # ["96","point","up"]
    idx = parts[0]            # 96
    direction = parts[-1]     # up / down

    # 获取所有子文件夹
    subfolders = [f for f in os.listdir(img_dir) if os.path.isdir(os.path.join(img_dir, f))]

    # 找编号对应的文件夹（只关心开头编号和最后的left/right）
    img_folder_left = next(
        (os.path.join(img_dir, f) for f in subfolders if f.startswith(f"{idx}_") and f.endswith("left")),
        None
    )
    img_folder_right = next(
        (os.path.join(img_dir, f) for f in subfolders if f.startswith(f"{idx}_") and f.endswith("right")),
        None
    )

    if not img_folder_left or not img_folder_right:
        print(f"⚠️ {file} 对应的图片文件夹缺失，跳过")
        continue

    # 左边预测
    df = process_images_for_csv(df, img_folder_left, "parking_left", model)
    # 右边预测
    df = process_images_for_csv(df, img_folder_right, "parking_right", model)

    # 保存结果
    out_csv = os.path.join(output_dir, file)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 完成 {file} -> {out_csv}")

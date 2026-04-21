import json
import os
from glob import glob

# 参数设置
json_dir = "JSON" 
label_output_dir = "TXT"
os.makedirs(label_output_dir, exist_ok=True)

# 类别映射（按需修改）
label_map = {
    "parallel": 0,
    "vertical": 1,
}

def convert_polygon_to_bbox(points):
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    xmin = min(x_coords)
    xmax = max(x_coords)
    ymin = min(y_coords)
    ymax = max(y_coords)
    return xmin, ymin, xmax, ymax

# 处理所有JSON文件
for json_file in glob(os.path.join(json_dir, "*.json")):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_path = os.path.basename(data["imagePath"]).replace("\\", "/")  # 只保留名字与拓展名，去掉路径
    image_name = os.path.splitext(image_path)[0]  # 只保留名字，不包含拓展名
    img_width = data["imageWidth"]
    img_height = data["imageHeight"]

    yolo_lines = []

    for shape in data["shapes"]:
        label = shape["label"]
        if label not in label_map:
            continue

        points = shape["points"]
        xmin, ymin, xmax, ymax = convert_polygon_to_bbox(points)

        # 中心点坐标与宽高（归一化）
        x_center = ((xmin + xmax) / 2) / img_width
        y_center = ((ymin + ymax) / 2) / img_height
        width = (xmax - xmin) / img_width
        height = (ymax - ymin) / img_height

        class_id = label_map[label]
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        yolo_lines.append(yolo_line)

    # 写入YOLO格式TXT
    output_path = os.path.join(label_output_dir, f"{image_name}.txt")
    with open(output_path, "w") as f:
        f.write("\n".join(yolo_lines))

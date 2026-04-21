import os, glob, re, time, csv, traceback, json, requests
from pathlib import Path

# ---------------- 工具函数 ----------------
def read_csv(filepath):
    data = []
    with open(filepath, newline='', encoding='utf-8') as f:
        for row in csv.reader(f):
            data.append(row)
    return data

def write_csv(filepath, data, head=None):
    if head:
        data = [head] + data
    with open(filepath, mode='w', encoding='UTF-8-sig', newline='') as f:
        csv.writer(f).writerows(data)

def wgs2bd09mc(wgs_x, wgs_y):
    """WGS84 → 百度墨卡托"""
    ak = ''  # Customize
    url = f'http://api.map.baidu.com/geoconv/v1/?coords={wgs_x},{wgs_y}&from=1&to=6&output=json&ak={ak}'
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).json()
    if res.get('status') == 0:
        return res['result'][0]['x'], res['result'][0]['y']
    return None, None

def getPanoId(bd09mc_x, bd09mc_y):
    """根据百度墨卡托坐标获取 svid"""
    url = f"https://mapsv0.bdimg.com/?qt=qsdata&x={bd09mc_x}&y={bd09mc_y}&l=17&action=0&mode=day&t={int(time.time()*1000)}"
    rsp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}).text
    m = re.search(r'"id":"(.+?)"', rsp)
    return m.group(1) if m else None

def grab_img_baidu(url):
    headers = {
        "Referer": "https://map.baidu.com/",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.212 Safari/537.36"
    }
    r = requests.get(url, headers=headers)
    return r.content if r.status_code == 200 and r.headers.get('Content-Type') == 'image/jpeg' else None

def normalize_deg(deg):
    """把角度归到 0-360，360 仍保留 360"""
    while deg < 0:
        deg += 360
    while deg > 360:
        deg -= 360
    return deg

def extract_number(name):
    m = re.search(r'(\d+)', Path(name).stem)
    return m.group(1) if m else None

def extract_angle_from_dirname(dirname):
    parts = Path(dirname).stem.split('_')
    if len(parts) >= 2:
        try:
            return normalize_deg(int(parts[1]))
        except ValueError:
            return None
    return None

# ---------------- 核心下载逻辑 ----------------
def download_one_csv(csv_path, root):
    csv_num = extract_number(csv_path.name)
    if not csv_num:
        print(f'⚠️ 跳过（无法提取编号）: {csv_path.name}')
        return

    # 找到 left/right 文件夹
    left_dirs = glob.glob(os.path.join(root, f'{csv_num}_*_left'))
    right_dirs = glob.glob(os.path.join(root, f'{csv_num}_*_right'))
    dirs = left_dirs + right_dirs
    if not dirs:
        print(f'⚠️ 未找到 {csv_num} 的 left/right 文件夹，跳过')
        return

    data = read_csv(csv_path)
    if not data:
        return
    header, rows = data[0], data[1:]

    error_list = []
    for dir_path in dirs:
        angle = extract_angle_from_dirname(dir_path)
        if angle is None:
            continue
        print(f'📂 处理目录: {os.path.basename(dir_path)}  angle={angle}')
        exist_imgs = set(glob.glob1(dir_path, '*.jpg'))

        for i, row in enumerate(rows, 1):
            try:
                wgs_x, wgs_y = row[0].strip(), row[1].strip()
                bd09mc_x, bd09mc_y = wgs2bd09mc(wgs_x, wgs_y)
                if bd09mc_x is None:
                    error_list.append([wgs_x, wgs_y, angle])
                    continue
                svid = getPanoId(bd09mc_x, bd09mc_y)
                if not svid:
                    error_list.append([wgs_x, wgs_y, angle])
                    continue

                img_fn = f'{wgs_x}_{wgs_y}_{angle}_0.jpg'
                if img_fn in exist_imgs:
                    continue

                url = f'https://mapsv0.bdimg.com/?qt=pr3d&fovy=90&quality=100&panoid={svid}&heading={angle}&pitch=0&width=800&height=600'
                img_bin = grab_img_baidu(url)
                if img_bin:
                    with open(os.path.join(dir_path, img_fn), 'wb') as f:
                        f.write(img_bin)
                    print(f'  ✅ {img_fn}')
                else:
                    print(f'  ❌ 下载失败: {img_fn}')
                    error_list.append([wgs_x, wgs_y, angle])
                time.sleep(6)   # 防封
            except Exception as e:
                print(f'  ⚠️ 异常: {e}')
                error_list.append([row[0], row[1], angle])

    if error_list:
        error_fn = os.path.join(root, 'error_nj.txt')
        with open(error_fn, 'a', encoding='utf-8') as f:
            for row in error_list:
                f.write(','.join(map(str, row)) + '\n')

# ---------------- 批量入口 ----------------
def process_all(root):
    for csv_path in Path(root).rglob('*.csv'):   # rglob:递归遍历所有子目录的.csv,.csv中采电坐标
        print(f'\n========== {csv_path.name} ==========')
        download_one_csv(csv_path, root)

if __name__ == '__main__':
    root = 'Result_predict\03_Gulou'
    process_all(root)

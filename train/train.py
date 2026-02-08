import os
import shutil
from roboflow import Roboflow
from ultralytics import YOLO
import multiprocessing
import torch

# =================設定區=================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MERGED_PATH = os.path.join(BASE_DIR, "datasets", "merged_dataset")
API_KEY = "Ll19suboP3L4pGTTxGjW"

# 資料集清單
DATASETS_LIST = [
    {"ws": "southern-lehigh-high-school", "proj": "frc-2026-yellow-balls", "ver": 1},
    {"ws": "robot-thing-qckxr", "proj": "frc-2026-game-pieces", "ver": 1},
    {"ws": "frcroboraiders", "proj": "frc-2026-fuel-sbrdk", "ver": 1},
    {"ws": "-wrw23", "proj": "frc-2026-fuel", "ver": 1},
    {"ws": "wayx112", "proj": "fuel-frc-256-do5cb", "ver": 1},
    {"ws": "myworkspace-mliyg", "proj": "frc-2026-rebuilt-fuel-detection", "ver": 1},
    {"ws": "workspace-8kf1w", "proj": "frc-2026-fuel-detection", "ver": 6},
    {"ws": "workspace-8kf1w", "proj": "frc-2026-fuel-detection", "ver": 7},
    {"ws": "workspace-8kf1w", "proj": "frc-2026-fuel-detection", "ver": 10}, 
    {"ws": "frc-photovision-test", "proj": "frc-project", "ver": 2},
    {"ws": "frc-photovision-test", "proj": "frc-project", "ver": 1}
]
# ========================================

def setup_directories():
    if os.path.exists(MERGED_PATH):
        print(f"🧹 清除舊資料: {MERGED_PATH}")
        shutil.rmtree(MERGED_PATH)
    for split in ['train', 'valid', 'test']:
        os.makedirs(os.path.join(MERGED_PATH, split, "images"), exist_ok=True)
        os.makedirs(os.path.join(MERGED_PATH, split, "labels"), exist_ok=True)

def download_and_merge():
    rf = Roboflow(api_key=API_KEY)
    downloaded_keys = set()
    print(f"📋 準備處理 {len(DATASETS_LIST)} 個資料集來源...")
    for i, d in enumerate(DATASETS_LIST):
        unique_key = f"{d['ws']}/{d['proj']}/v{d['ver']}"
        if unique_key in downloaded_keys: continue
        try:
            print(f"\n📦 [{i+1}/{len(DATASETS_LIST)}] 下載中: {unique_key} ...")
            ds = rf.workspace(d['ws']).project(d['proj']).version(d['ver']).download("yolov8")
            for split in ['train', 'valid', 'test']:
                src_img_dir = os.path.join(ds.location, split, "images")
                src_lbl_dir = os.path.join(ds.location, split, "labels")
                target_img_dir = os.path.join(MERGED_PATH, split, "images")
                target_lbl_dir = os.path.join(MERGED_PATH, split, "labels")
                if os.path.exists(src_img_dir):
                    files = os.listdir(src_img_dir)
                    for file in files:
                        if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')): continue
                        new_filename = f"set{i}_{file}"
                        shutil.copy(os.path.join(src_img_dir, file), os.path.join(target_img_dir, new_filename))
                        label_file = file.rsplit('.', 1)[0] + ".txt"
                        src_txt_path = os.path.join(src_lbl_dir, label_file)
                        if os.path.exists(src_txt_path):
                            shutil.copy(src_txt_path, os.path.join(target_lbl_dir, new_filename.rsplit('.', 1)[0] + ".txt"))
            downloaded_keys.add(unique_key)
        except Exception as e:
            print(f"⚠️ 下載失敗 [{unique_key}]: {e}")

def create_yaml():
    yaml_content = f"train: {os.path.join(MERGED_PATH, 'train', 'images')}\nval: {os.path.join(MERGED_PATH, 'valid', 'images')}\ntest: {os.path.join(MERGED_PATH, 'test', 'images')}\nnc: 1\nnames: ['fuel']"
    yaml_path = os.path.join(MERGED_PATH, "data.yaml")
    with open(yaml_path, 'w') as f: f.write(yaml_content)
    return yaml_path

def main():
    setup_directories()
    download_and_merge()
    yaml_path = create_yaml()

    print("🚀 開始訓練 YOLOv8 模型 (Local Pip版)...")
    
    # 直接載入模型，device=0 會自動找第一張顯卡
    model = YOLO("yolov8n.pt") 

    results = model.train(
        data=yaml_path,
        epochs=80,
        imgsz=320,
        batch=16,       # 若顯存(VRAM)大於 8GB 可改成 32; 若報錯 OOM 改成 8
        patience=20,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        mosaic=1.0,
        exist_ok=True,
        workers=4,      # Windows 上如果訓練時卡住不動，請改為 0 或 1
        device=0        # 強制指定 GPU
    )
    
    print("🎉 訓練結束！模型已儲存至 runs/detect/train")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
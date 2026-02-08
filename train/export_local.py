from ultralytics import YOLO

# 載入你剛練好的模型
model = YOLO("runs/detect/train/weights/best.pt")

# 導出 INT8 TFLite
# ⚠️ 這裡的 data='data.yaml' 很重要，它會去讀你的訓練圖片來校正精度
# ⚠️ imgsz=320 必須跟訓練時一樣
print("正在導出 INT8 TFLite (這可能需要幾分鐘)...")
model.export(format="tflite", int8=True, data="data.yaml", imgsz=320)

print("✅ 完成！請在資料夾中找到 'best_int8.tflite' (或是 best_full_integer_quant.tflite)")
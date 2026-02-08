//訓練模型方法//

建立虛擬環境
python -m venv venv

啟動環境
venv\Scripts\activate

安裝支援 N 卡的 PyTorch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

安裝 YOLO 和 Roboflow
pip install ultralytics roboflow

確認顯卡是否抓到
python -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}')"

執行方式
python train.py

python export_local.py

去 Google Colab 轉成 TPU 格式

開啟一個新的 Google Colab 筆記本。

不需要 開啟 GPU 模式 (我們只是要編譯，CPU 就夠了)。

把剛剛電腦上的 best.pt 拖曳上傳 到 Colab 左邊的檔案區。

複製以下程式碼到 Colab 的單元格並執行：

# 1. 安裝必要套件
!pip install ultralytics
!curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
!echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
!sudo apt-get update
!sudo apt-get install edgetpu-compiler

# 2. 開始轉檔 (PT -> TFLite INT8)
from ultralytics import YOLO

# 載入你上傳的模型
model = YOLO("best.pt")

# 直接導出 TFLite (在 Colab 轉檔不一定要 data.yaml，它會自動進行全整數校正)
# 這裡建議使用 imgsz=320，跟原本一致
model.export(format="tflite", int8=True, imgsz=320)

# 3. 編譯為 Edge TPU 格式
import os

# 尋找剛產生的 tflite 檔案路徑
# 通常在目前的目錄下
tflite_path = "best_saved_model/best_full_integer_quant.tflite"

if os.path.exists(tflite_path):
    print("🚀 發現 TFLite 檔案，開始編譯為 Edge TPU 格式...")
    !edgetpu_compiler -s {tflite_path}
    print("✅ 編譯完成！請下載 _edgetpu.tflite 檔案")
else:
    # 有時候路徑會在別的地方，列出檔案確認
    !find . -name "*.tflite"
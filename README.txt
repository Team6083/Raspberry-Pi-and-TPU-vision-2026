
📦 硬體準備

Raspberry Pi 4 

Google Coral USB Accelerator (TPU)

USB 攝影機 (或 Pi Camera)

MicroSD 卡 (建議 32GB 以上)


第二階段：環境建置

📦安裝 64-bit 的作業系統

下載並安裝 Raspberry Pi Imager (在你的電腦上)

Raspberry Pi OS (other) -> Raspberry Pi OS (64-bit)

設定 (齒輪圖示 ⚙️) - 這步最重要！：

Hostname: Team6083

Username: cms-robotics

Password: 60836083(在進入程式時不會顯示出來)

Enable SSH: 打勾 (選 Use password authentication)

Wi-Fi: 設定你的熱點 (SSID 和密碼)，這樣開機就能連網。

第二階段：環境建置

📦 先把他們下載樹梅派:

================================================================
=  libedgetpu1-std_16.0tf2.17.1-1.trixie_arm64.deb             =
=  pycoral-2.0.3-cp312-cp312-linux_aarch64.whl                 =
=  tflite_runtime-2.17.1-cp312-cp312-linux_aarch64.whl         =
================================================================

//利用cmd進入樹梅派
  ssh cms-robotics@10.141.3.XX(要去找)

  輸入密碼:60836083

📦 下載python3.12

sudo apt update
sudo apt install python3.12 python3.12-venv

📦 檢查python版本

python3.12 --version

📦 安裝 Conda
 
# 下載安裝腳本
wget https://github.com/conda- forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
 
# 執行安裝 (一路按 Enter，最後問 init 選 yes)
bash Miniforge3-Linux-aarch64.sh

# 讓設定生效
source ~/.bashrc

📦 建立 Python 3.12 環境

# 1. 建立名為 robot_env 的環境，指定 python 3.12
conda create -n robot_env python=3.12 -y

# 2. 啟動環境
conda activate robot_env

📦 在 Conda 內安裝

# 0. 安裝 Edge TPU 驅動
sudo apt update
sudo apt install ./libedgetpu1-std_16.0tf2.17.1-1.trixie_arm64.deb

# 1. 先升級 pip (避免安裝失敗)
pip install --upgrade pip

# 2. 安裝 TFLite Runtime (這是 PyCoral 的基礎)
pip install tflite_runtime-2.17.1-cp312-cp312-linux_aarch64.whl

# 3. 安裝 PyCoral
pip install pycoral-2.0.3-cp312-cp312-linux_aarch64.whl

# 4. 建立資料夾 
mkdir coral_test

# 5. 建立python檔
cd. > detect.py

存檔離開 (Ctrl+O -> Enter -> Ctrl+X)

📦 安裝其他 FRC 必備套件

# 安裝 OpenCV (不含 GUI 版，比較輕量)、Flask、NumPy
pip install opencv-python-headless flask numpy

# 安裝 NetworkTables (RobotRIO用)
pip install robotpy-ntcore


//目前初步完成環境設置//

📦 一般開機進入環境 

輸入指令 

進入環境
conda activate robot_env

進入資料夾 
cd coral_test/

# 1. 編輯檔案 
nano detect.py

# 2. 開啟檔案
python detect.py

//建立開機服務//

建立一個 .service 檔案
sudo nano /etc/systemd/system/frc_vision.service

編輯內容
[Unit]
Description=FRC 6083 Vision Service
# 確保網路連線後才啟動 (FRC 機器人需要連 NetworkTables)
After=network.target

[Service]
# 設定使用者 (非常重要，不然會讀不到檔案)
User=cms-robotics
Group=cms-robotics

# 設定工作目錄 (你的程式在哪裡)
WorkingDirectory=/home/cms-robotics/coral_test

# 啟動指令 (直接用 Conda 環境的 Python 執行)
# 這裡用的是絕對路徑，確保不會抓錯 Python
ExecStart=/home/cms-robotics/miniforge3/envs/robot_env/bin/python detect.py

# 如果程式當掉，自動重啟
Restart=always
# 當掉後等待 5 秒再重啟
RestartSec=5

# 輸出 Log 設定 (方便除錯)
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target

存檔離開 (Ctrl+O -> Enter -> Ctrl+X)

重整系統設定
sudo systemctl daemon-reload

設定開機自動啟動
sudo systemctl enable frc_vision.service

立刻啟動服務
sudo systemctl start frc_vision.service

停止服務
sudo systemctl stop frc_vision.service

啟動服務
sudo systemctl start frc_vision.service

重啟服務
sudo systemctl restart frc_vision.service


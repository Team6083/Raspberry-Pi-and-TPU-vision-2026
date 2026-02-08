import cv2
import numpy as np
import time
import threading
import subprocess
import os
import csv  # 🔥【新增】CSV 模組
from datetime import datetime
import ntcore
from flask import Flask, Response, render_template_string, jsonify
from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import common

# ==================== 參數設定 ====================
MODEL_PATH = "best_full_integer_quant_edgetpu_4.tflite"
TEAM_NUMBER = 6083
EXPOSURE_VAL = 30
BRIGHTNESS_VAL = 90
ALPHA = 0.2
CALIB_X_OFFSET = 10 
CALIB_Y_OFFSET = -10
CALIB_SCALE = 1.1

VIDEO_DIR = "captured_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)
# =================================================

app = Flask(__name__)
output_frame = None
lock = threading.Lock()

s_x, s_w = 0, 0
current_ball_count = 0
current_target_x, current_target_y = 0.0, 0.0
detection_counter = 0
ENABLE_NT = False
current_brightness = BRIGHTNESS_VAL
brightness_changed = False
last_target_center = None

# 🔥【修改】錄影相關變數
is_recording = False
video_writer = None
csv_file = None   # 🔥 CSV 檔案物件
csv_writer = None # 🔥 CSV 寫入器
frame_idx = 0     # 🔥 幀數計數器

class TFLiteTranslator:
    def __init__(self, model_path):
        self.interpreter = make_interpreter(model_path)
        self.interpreter.allocate_tensors()
        self.output_details = self.interpreter.get_output_details()[0]
        self.o_scale, self.o_zp = self.output_details['quantization']

    def process(self, letterbox_img):
        img = cv2.cvtColor(letterbox_img, cv2.COLOR_BGR2RGB)
        common.set_input(self.interpreter, img)
        self.interpreter.invoke()
        raw_output = self.interpreter.get_tensor(self.output_details['index'])[0]
        data = (raw_output.astype(np.float32) - self.o_zp) * self.o_scale
        return data.T if data.shape[0] == 5 else data

def vision_worker():
    global output_frame, s_x, s_w, detection_counter, current_ball_count
    global current_target_x, current_target_y, current_brightness, brightness_changed, last_target_center
    global is_recording, video_writer, csv_file, csv_writer, frame_idx # 🔥 引用變數
    
    translator = TFLiteTranslator(MODEL_PATH)
    cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if not cap.isOpened(): cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_VAL)
    cap.set(cv2.CAP_PROP_BRIGHTNESS, current_brightness)
    cap.set(cv2.CAP_PROP_GAIN, 0)

    # NT 初始化 (省略...)
    if ENABLE_NT:
        inst = ntcore.NetworkTableInstance.getDefault()
        table = inst.getTable("6083_Vision")
        x_pub = table.getDoubleTopic("closest_x").publish()
        y_pub = table.getDoubleTopic("closest_dist").publish()
        count_pub = table.getIntegerTopic("ball_count").publish()
        inst.setServerTeam(TEAM_NUMBER)
        inst.startClient4("RPi_6083_Calibrated")

    def terminal_display():
        while True:
            time.sleep(1.0)
    threading.Thread(target=terminal_display, daemon=True).start()

    while True:
        if brightness_changed:
            cap.set(cv2.CAP_PROP_BRIGHTNESS, current_brightness)
            brightness_changed = False
        
        ret, frame = cap.read()
        if not ret: continue

        # ==========================================
        # 🔥🔥🔥 錄影與 CSV 初始化 🔥🔥🔥
        # ==========================================
        if is_recording:
            if video_writer is None:
                # 檔名時間戳記
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                # 1. 建立影片
                vid_filename = os.path.join(VIDEO_DIR, f"{timestamp}.avi")
                fourcc = cv2.VideoWriter_fourcc(*'MJPG') 
                video_writer = cv2.VideoWriter(vid_filename, fourcc, 30.0, (640, 480))
                
                # 2. 建立 CSV (同檔名，不同副檔名)
                csv_filename = os.path.join(VIDEO_DIR, f"{timestamp}.csv")
                csv_file = open(csv_filename, 'w', newline='')
                csv_writer = csv.writer(csv_file)
                # 寫入標題列：幀數, X座標, Y座標, 寬, 高, 信心度
                csv_writer.writerow(["Frame", "X", "Y", "W", "H", "Confidence"])
                
                frame_idx = 0 # 重置計數
                print(f"🔴 開始錄影: {vid_filename}")
            
            # 寫入乾淨影像
            video_writer.write(frame)
            frame_idx += 1
        else:
            # 停止錄影與關閉檔案
            if video_writer is not None:
                video_writer.release()
                video_writer = None
                csv_file.close() # 關閉 CSV
                csv_file = None
                print("⏹ 停止錄影，檔案已儲存。")
        
        # ==========================================
        #  影像辨識流程
        # ==========================================
        resized_240 = cv2.resize(frame, (320, 240))
        input_canvas = np.zeros((320, 320, 3), dtype=np.uint8)
        input_canvas[40:280, 0:320] = resized_240

        data = translator.process(input_canvas)
        boxes, confs = [], []
        
        mask = data[:, 4] > 0.35 
        for row in data[mask]:
            xc, yc, w, h, conf = row
            # 還原與校正...
            real_x = xc * 640 + CALIB_X_OFFSET
            real_y = ((yc * 320 - 40) / 240) * 480 + CALIB_Y_OFFSET
            real_w = w * 640 * CALIB_SCALE
            real_h = (h * 320 / 240) * 480 * CALIB_SCALE
            
            # 計算左上角座標
            x1, y1 = int(real_x - real_w/2), int(real_y - real_h/2)
            w_int, h_int = int(real_w), int(real_h)

            boxes.append([x1, y1, w_int, h_int])
            confs.append(float(conf))

        indices = cv2.dnn.NMSBoxes(boxes, confs, 0.35, 0.45)
        
        # ==========================================
        # 🔥🔥🔥 將辨識結果寫入 CSV 🔥🔥🔥
        # ==========================================
        if is_recording and csv_writer is not None:
            if len(indices) > 0:
                for i in indices.flatten():
                    b = boxes[i]
                    c = confs[i]
                    # 格式: Frame, X, Y, W, H, Conf
                    csv_writer.writerow([frame_idx, b[0], b[1], b[2], b[3], f"{c:.2f}"])
            else:
                # 這一幀沒抓到東西，寫入空紀錄 (方便除錯)
                csv_writer.writerow([frame_idx, -1, -1, -1, -1, 0])

        # ==========================================
        #  後續邏輯 (選目標、畫圖) 保持不變
        # ==========================================
        if len(indices) > 0:
            detection_counter += 1
            if detection_counter >= 2:
                current_ball_count = len(indices)
                all_valid_boxes = [boxes[i] for i in indices.flatten()]
                all_confs = [confs[i] for i in indices.flatten()]
                
                # ... (原有的選目標邏輯) ...
                best_score = -999
                target_box = None
                for i, b in enumerate(all_valid_boxes):
                     # ... (計算分數) ...
                     y_score = (b[1] + b[3]) / 480.0
                     conf_score = all_confs[i]
                     total_score = (y_score * 0.7) + (conf_score * 0.3)
                     if total_score > best_score:
                        best_score = total_score
                        target_box = b

                if target_box:
                    last_target_center = (target_box[0] + target_box[2]/2, target_box[1] + target_box[3]/2)
                    s_x = int(ALPHA * target_box[0] + (1 - ALPHA) * s_x)
                    s_w = int(ALPHA * target_box[2] + (1 - ALPHA) * s_w)
                    current_target_x = ((s_x + s_w/2) / 640.0) * 2 - 1
                    current_target_y = (target_box[1] + target_box[3]) / 480.0
                    
                    if ENABLE_NT:
                        x_pub.set(current_target_x)
                        y_pub.set(current_target_y)
                        count_pub.set(current_ball_count)

                    for b in all_valid_boxes:
                        is_target = (b == target_box)
                        color = (0, 255, 0) if is_target else (100, 100, 100)
                        cv2.rectangle(frame, (b[0], b[1]), (b[0]+b[2], b[1]+b[3]), color, 3 if is_target else 1)
                    
                    # 畫中心點
                    target_cx = int(s_x + s_w/2)
                    target_cy = int(target_box[1] + target_box[3]/2)
                    cv2.circle(frame, (target_cx, target_cy), 5, (0, 0, 255), -1)
                    cv2.line(frame, (320, 240), (target_cx, target_cy), (0, 255, 255), 2)
                    cv2.putText(frame, f"OFFSET {current_target_x:.2f}", (s_x, target_box[1]-10), 0, 0.7, (0, 255, 0), 2)

        else:
            detection_counter = 0
            current_ball_count = 0
            if ENABLE_NT: count_pub.set(0)

        if is_recording:
             cv2.circle(frame, (620, 20), 10, (0, 0, 255), -1)

        with lock:
            output_frame = frame.copy()

# --- Flask 部分不需要動 ---
HTML_TEMPLATE = '''
<html>
<head><style>
    body { background: #1a1a1a; color: white; text-align: center; font-family: monospace; }
    .stat-box { background: #333; padding: 15px; margin: 10px; border-radius: 12px; display: inline-block; min-width: 150px; border: 1px solid #555; vertical-align: top; }
    .val { font-size: 32px; color: #ffff00; font-weight: bold; margin: 5px 0; }
    .label { color: #aaa; font-size: 14px; text-transform: uppercase; }
    .btn { background: #444; color: white; border: 1px solid #777; padding: 5px 15px; font-size: 18px; border-radius: 5px; cursor: pointer; margin: 0 5px; }
    .btn:hover { background: #666; } .btn:active { background: #00ff00; color: black; }
    
    /* 錄影按鈕樣式 */
    .btn-rec { background: #27ae60; width: 200px; padding: 10px; font-weight: bold; margin-top: 10px; border-radius: 8px;}
    .recording { background: #c0392b; animation: pulse 1s infinite; }
    @keyframes pulse { 0% { opacity: 1; } 50% { opacity: 0.7; } 100% { opacity: 1; } }
    
    .connected { color: #00ff00; } .local { color: #ff4444; }
</style>
<script>
    function adjustBrightness(val) {
        fetch('/adjust_brightness/' + val).then(r => r.json()).then(data => {
            document.getElementById('bright_val').innerText = data.new_brightness;
        });
    }

    // 錄影控制
    function toggleRecord() {
        fetch('/toggle_record').then(r => r.json()).then(data => {
            updateRecButton(data.is_recording);
        });
    }

    function updateRecButton(isRec) {
        let btn = document.getElementById('recBtn');
        if (isRec) {
            btn.innerText = "⏹ 停止錄影 (REC)";
            btn.classList.add("recording");
        } else {
            btn.innerText = "🔴 開始蒐集資料 (Start)";
            btn.classList.remove("recording");
        }
    }

    setInterval(function(){
        fetch('/get_status').then(r => r.json()).then(data => {
            document.getElementById('count').innerText = data.count;
            document.getElementById('tx').innerText = data.tx.toFixed(2);
            document.getElementById('ty').innerText = data.ty.toFixed(2);
            // 同步錄影狀態
            updateRecButton(data.is_recording);
        });
    }, 500);
</script>
</head>
<body>
    <h1 style="color: #00ff00; text-shadow: 0 0 10px #00ff00;">6083 INTAKE DASHBOARD</h1>
    
    <div class="stat-box">
        <div class="label">當前亮度</div>
        <div id="bright_val" class="val">{{ br }}</div>
        <button class="btn" onclick="adjustBrightness(-10)">- 10</button>
        <button class="btn" onclick="adjustBrightness(10)">+ 10</button>
    </div>

    <div class="stat-box">
        <div class="label">場上球數</div>
        <div id="count" class="val">0</div>
    </div>

    <div class="stat-box">
        <div class="label">目標 X 偏移</div>
        <div id="tx" class="val">0.00</div>
    </div>
    
    <div class="stat-box">
        <div class="label">目標距離 (Y)</div>
        <div id="ty" class="val" style="color: #00ff00;">0.00</div>
    </div>

    <br>
    <button id="recBtn" class="btn btn-rec" onclick="toggleRecord()">🔴 開始蒐集資料 (Start)</button>

    <div style="margin-top: 10px;">
        <span class="{{ 'connected' if nt else 'local' }}" style="font-weight:bold; font-size: 18px;">
            {{ '● NETWORKTABLES CONNECTED' if nt else '● LOCAL MODE' }}
        </span>
    </div>
    <img src="/video_feed" width="640" style="border: 3px solid #555; border-radius: 10px; margin-top: 15px; box-shadow: 0 0 20px rgba(0,255,0,0.1);">
</body>
</html>
'''

@app.route('/')
def index(): 
    return render_template_string(HTML_TEMPLATE, nt=ENABLE_NT, br=current_brightness)

# 🔥【新增】錄影控制 API
@app.route('/toggle_record')
def toggle_record():
    global is_recording
    is_recording = not is_recording
    return jsonify({"is_recording": is_recording})

@app.route('/adjust_brightness/<int:val>')
def adjust_brightness(val):
    global current_brightness, brightness_changed
    current_brightness = max(0, min(255, current_brightness + val))
    brightness_changed = True
    return jsonify({"new_brightness": current_brightness})

@app.route('/get_status')
def get_status():
    # 回傳狀態也包含錄影資訊
    return jsonify({
        "count": current_ball_count, 
        "tx": current_target_x, 
        "ty": current_target_y,
        "is_recording": is_recording
    })

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with lock:
                if output_frame is None: continue
                ret, b = cv2.imencode('.jpg', output_frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                f_bytes = b.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + f_bytes + b'\r\n')
            time.sleep(0.04)
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # 自動偵測 RoboRIO
    ROBORIO_IP = "10.60.83.2"
    print(f"正在 Ping 偵測 RoboRIO ({ROBORIO_IP})...")
    response = os.system(f"ping -c 1 -W 1 {ROBORIO_IP} > /dev/null 2>&1")

    if response == 0:
        print("✅ 成功連線到 RoboRIO！自動啟動 NetworkTables 模式。")
        ENABLE_NT = True
    else:
        print("⚠️ 找不到 RoboRIO。自動進入「單機測試模式」。")
        ENABLE_NT = False

    subprocess.run(["sudo", "fuser", "-k", "5000/tcp"], capture_output=True)
    threading.Thread(target=vision_worker, daemon=True).start()
    app.run(host='0.0.0.0', port=5000, threaded=True, debug=False, use_reloader=False)
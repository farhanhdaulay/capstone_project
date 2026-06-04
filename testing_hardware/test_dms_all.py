#!/usr/bin/env python3
"""
test_dms_all.py  —  DMS Unified Test
Tests: Head Pose (6DRepNet) + EAR + MAR (PFLD) + Phone Detection (YOLOv11n)
Camera: IMX219 via GStreamer  |  Jetson Orin Nano Super
Run:  cd ~/dms_project && python3 tests/test_dms_all.py
View: http://<jetson-ip>:5000
"""

import os, sys, time, cv2, numpy as np, threading
from flask import Flask, Response

# ── No DISPLAY needed — we stream via browser ────────────────────────────────
os.environ.pop("DISPLAY", None)

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))
MODEL_DIR    = os.path.join(PROJECT_ROOT, "models")

# ────────────────────────────────────────────────────────────────────────────
# 1.  CAMERA
# ────────────────────────────────────────────────────────────────────────────
def open_camera(source="csi", width=640, height=480, fps=30):
    if source == "csi":
        pipeline = (
            f"nvarguscamerasrc ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"framerate={fps}/1, format=NV12 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! "
            f"appsink"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            raise RuntimeError(f"Camera failed to open (source={source})")
        for _ in range(5):
            ret, frame = cap.read()
            if ret and frame is not None:
                break
        else:
            cap.release()
            raise RuntimeError("Camera opened but no frames received.\n"
                               f"Pipeline: {pipeline}")
    else:
        cap = cv2.VideoCapture(int(source))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS,          fps)
        if not cap.isOpened():
            raise RuntimeError(f"Camera failed to open (source={source})")

    print(f"[OK ] Camera opened  (source={source}, {width}x{height} @ {fps}fps)")
    return cap


# ────────────────────────────────────────────────────────────────────────────
# 2.  PFLD  —  EAR + MAR
# ────────────────────────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
    ONNX_OK = True
except ImportError:
    ONNX_OK = False
    print("[WARN] onnxruntime not found — PFLD + 6DRepNet will be skipped")

# 68-point indices for pfld_68
LEFT_EYE  = [36, 37, 38, 39, 40, 41]
RIGHT_EYE = [42, 43, 44, 45, 46, 47]
MOUTH_TOP = 51
MOUTH_BOT = 57
MOUTH_L   = 48
MOUTH_R   = 54

def _euc(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))

def compute_EAR(pts, idx):
    p = [pts[i] for i in idx]
    return ((_euc(p[1], p[5]) + _euc(p[2], p[4])) /
            (2.0 * _euc(p[0], p[3]) + 1e-6))

def compute_MAR(pts):
    return (_euc(pts[MOUTH_TOP], pts[MOUTH_BOT]) /
            (_euc(pts[MOUTH_L],  pts[MOUTH_R])   + 1e-6))


class PFLDTester:
    INPUT_SIZE = (112, 112)

    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"PFLD model not found: {model_path}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess     = ort.InferenceSession(model_path, providers=providers)
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        print(f"[OK ] PFLD loaded  ({os.path.basename(model_path)})")
        print(f"      Provider: {self.sess.get_providers()[0]}")

    def _face_roi(self, frame):
        h, w = frame.shape[:2]
        m = 0.15
        x1, y1 = int(w * m),       int(h * m)
        x2, y2 = int(w * (1 - m)), int(h * (1 - m))
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def run(self, frame):
        roi, bbox = self._face_roi(frame)
        if roi.size == 0:
            return None, None, bbox, [], roi
        img  = cv2.resize(roi, self.INPUT_SIZE)
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp  = img.transpose(2, 0, 1)[np.newaxis, :]
        raw  = self.sess.run([self.out_name], {self.in_name: inp})[0][0]
        pts  = [
            (float(raw[i * 2])     * roi.shape[1],
             float(raw[i * 2 + 1]) * roi.shape[0])
            for i in range(68)
        ]
        ear = (compute_EAR(pts, LEFT_EYE) + compute_EAR(pts, RIGHT_EYE)) / 2.0
        mar = compute_MAR(pts)
        return round(ear, 3), round(mar, 3), bbox, pts, roi


# ────────────────────────────────────────────────────────────────────────────
# 3.  HEAD POSE  —  6DRepNet360
# ────────────────────────────────────────────────────────────────────────────
class HeadPoseTester:
    INPUT_SIZE = (224, 224)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"6DRepNet model not found: {model_path}")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess     = ort.InferenceSession(model_path, providers=providers)
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name
        print(f"[OK ] 6DRepNet loaded  ({os.path.basename(model_path)})")
        print(f"      Provider: {self.sess.get_providers()[0]}")

    def run(self, frame, face_bbox=None):
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            roi = frame[y1:y2, x1:x2]
        else:
            h, w = frame.shape[:2]
            m    = int(min(h, w) * 0.15)
            roi  = frame[m:h - m, m:w - m]
        if roi.size == 0:
            return 0.0, 0.0, 0.0
    
        img = cv2.resize(roi, self.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        inp = img.transpose(2, 0, 1)[np.newaxis, :]
    
        # Output is rotation matrix (1, 3, 3)
        R = self.sess.run([self.out_name], {self.in_name: inp})[0][0]  # shape (3,3)
    
        # Convert rotation matrix to pitch, yaw, roll (degrees)
        pitch = float(np.degrees(np.arcsin(-R[2, 0])))
        yaw   = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
        roll  = float(np.degrees(np.arctan2(R[2, 1], R[2, 2])))
    
        return round(pitch, 1), round(yaw, 1), round(roll, 1)


# ────────────────────────────────────────────────────────────────────────────
# 4.  PHONE DETECTION  —  YOLOv11n
# ────────────────────────────────────────────────────────────────────────────
PHONE_CLASS = 67

class PhoneTester:
    CONF = 0.40

    def __init__(self, model_path):
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"YOLO model not found: {model_path}")
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        print(f"[OK ] YOLOv11n loaded  ({os.path.basename(model_path)})")

    def run(self, frame):
        results = self.model(frame, verbose=False, conf=self.CONF)
        phones  = []
        for box in results[0].boxes:
            if int(box.cls) == PHONE_CLASS:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                phones.append((x1, y1, x2, y2, float(box.conf)))
        return phones


# ────────────────────────────────────────────────────────────────────────────
# 5.  DRAW OVERLAY
# ────────────────────────────────────────────────────────────────────────────
def draw_overlay(frame, ear, mar, pitch, yaw, roll, phones, face_bbox):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    if face_bbox:
        x1, y1, x2, y2 = face_bbox
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 200, 255), 2)

    for px1, py1, px2, py2, conf in phones:
        cv2.rectangle(overlay, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.putText(overlay, f"PHONE {conf:.2f}",
                    (px1, py1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

    panel_y = 30
    def put(text, y, color=(220, 220, 220)):
        cv2.putText(overlay, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,0,0), 4)
        cv2.putText(overlay, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color,  2)

    ear_color = (0, 60, 255) if (ear is not None and ear < 0.20) else (0, 230, 80)
    put(f"EAR : {ear:.3f}" if ear is not None else "EAR : --", panel_y, ear_color)
    panel_y += 32

    mar_color = (0, 140, 255) if (mar is not None and mar > 0.60) else (0, 230, 80)
    put(f"MAR : {mar:.3f}" if mar is not None else "MAR : --", panel_y, mar_color)
    panel_y += 32

    yaw_alert   = abs(yaw)   > 30 if yaw   is not None else False
    pitch_alert = abs(pitch) > 30 if pitch is not None else False
    pose_color  = (0, 60, 255) if (yaw_alert or pitch_alert) else (0, 230, 80)
    put((f"Pitch:{pitch:+.1f}  Yaw:{yaw:+.1f}  Roll:{roll:+.1f}"
         if pitch is not None else "Head Pose: --"), panel_y, pose_color)
    panel_y += 32

    phone_color = (0, 60, 255) if phones else (0, 230, 80)
    put(f"Phone: {len(phones)} detected" if phones else "Phone: none",
        panel_y, phone_color)

    if ear is not None and ear < 0.20:
        state, sc = "DROWSY", (0, 60, 255)
    elif mar is not None and mar > 0.60:
        state, sc = "YAWNING", (0, 140, 255)
    elif yaw_alert or pitch_alert:
        state, sc = "DISTRACTED", (0, 100, 255)
    elif phones:
        state, sc = "PHONE DETECTED", (0, 60, 255)
    else:
        state, sc = "NORMAL", (0, 220, 80)

    cv2.rectangle(overlay, (0, h - 44), (w, h), (20, 20, 20), -1)
    cv2.putText(overlay, state, (w // 2 - 120, h - 12),
                cv2.FONT_HERSHEY_DUPLEX, 1.0, sc, 2)
    cv2.putText(overlay, "DMS Live  |  browser stream",
                (8, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1)

    return overlay


# ────────────────────────────────────────────────────────────────────────────
# 6.  FLASK MJPEG SERVER
# ────────────────────────────────────────────────────────────────────────────
app          = Flask(__name__)
output_frame = None
frame_lock   = threading.Lock()


def generate():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.01)
                continue
            ret, buf = cv2.imencode(".jpg", output_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            data = buf.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
        time.sleep(0.03)


@app.route("/")
def index():
    return """
    <html>
    <head>
      <title>DMS Live</title>
      <style>
        body { background:#111; display:flex; flex-direction:column;
               align-items:center; justify-content:center; height:100vh; margin:0; }
        h2   { color:#eee; font-family:monospace; margin-bottom:12px; }
        img  { border:2px solid #444; border-radius:6px; max-width:95vw; }
      </style>
    </head>
    <body>
      <h2>DMS Live Stream</h2>
      <img src="/video_feed">
    </body>
    </html>
    """


@app.route("/video_feed")
def video_feed():
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


# ────────────────────────────────────────────────────────────────────────────
# 7.  INFERENCE LOOP (runs in background thread)
# ────────────────────────────────────────────────────────────────────────────
def inference_loop(cap, pfld, pose, phone, width):
    global output_frame

    fps_t    = time.time()
    fps_cnt  = 0
    fps_disp = 0.0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue

        ear = mar = pitch = yaw = roll = None
        face_bbox = None
        pts_out   = None
        phones    = []

        if pfld:
            try:
                result = pfld.run(frame)
                if len(result) == 5:
                    ear, mar, face_bbox, pts_out, _ = result
                else:
                    ear, mar, face_bbox = result
            except Exception as e:
                print(f"[ERR ] PFLD: {e}")

        if pose:
            try:
                pitch, yaw, roll = pose.run(frame, face_bbox)
            except Exception as e:
                print(f"[ERR ] 6DRepNet: {e}")

        if phone:
            try:
                phones = phone.run(frame)
            except Exception as e:
                print(f"[ERR ] YOLO: {e}")

        if pts_out and face_bbox:
            ox, oy = face_bbox[0], face_bbox[1]
            for px, py in pts_out:
                cv2.circle(frame, (int(px + ox), int(py + oy)), 1, (0, 255, 180), -1)

        fps_cnt += 1
        if time.time() - fps_t >= 1.0:
            fps_disp = fps_cnt / (time.time() - fps_t)
            fps_cnt  = 0
            fps_t    = time.time()
            print(f"FPS={fps_disp:.1f}  EAR={ear}  MAR={mar}  "
                  f"Pitch={pitch}  Yaw={yaw}  Roll={roll}  "
                  f"Phones={len(phones)}")

        cv2.putText(frame, f"FPS {fps_disp:.1f}", (width - 110, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        vis = draw_overlay(frame, ear, mar, pitch, yaw, roll, phones, face_bbox)

        with frame_lock:
            output_frame = vis


# ────────────────────────────────────────────────────────────────────────────
# 8.  ENTRY POINT
# ────────────────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="DMS unified test — browser stream")
    parser.add_argument("--source",   default="csi")
    parser.add_argument("--width",    type=int, default=640)
    parser.add_argument("--height",   type=int, default=480)
    parser.add_argument("--fps",      type=int, default=30)
    parser.add_argument("--port",     type=int, default=5000)
    parser.add_argument("--no-pfld",  action="store_true")
    parser.add_argument("--no-pose",  action="store_true")
    parser.add_argument("--no-phone", action="store_true")
    args = parser.parse_args()

    # ── Camera first — before model loading ─────────────────────────────────
    cap = open_camera(args.source, args.width, args.height, args.fps)

    # ── Models ──────────────────────────────────────────────────────────────
    pfld  = None
    pose  = None
    phone = None

    if ONNX_OK and not args.no_pfld:
        try:
            pfld = PFLDTester(os.path.join(MODEL_DIR, "pfld_106_lite.onnx"))
        except Exception as e:
            print(f"[WARN] PFLD load failed: {e}")

    if ONNX_OK and not args.no_pose:
        try:
            pose = HeadPoseTester(os.path.join(MODEL_DIR, "6drepnet360.onnx"))
        except Exception as e:
            print(f"[WARN] 6DRepNet load failed: {e}")

    if not args.no_phone:
        try:
            yolo_engine = os.path.join(MODEL_DIR, "yolo11n.engine")
            yolo_pt     = os.path.join(MODEL_DIR, "yolo11n.pt")
            yolo_path   = yolo_engine if os.path.isfile(yolo_engine) else yolo_pt
            phone = PhoneTester(yolo_path)
        except Exception as e:
            print(f"[WARN] YOLO load failed: {e}")

    print("\n── Models ready ──────────────────────────────────────────────")
    print(f"   PFLD  EAR/MAR : {'YES' if pfld  else 'SKIPPED'}")
    print(f"   6DRepNet Pose : {'YES' if pose  else 'SKIPPED'}")
    print(f"   YOLOv11n Phone: {'YES' if phone else 'SKIPPED'}")
    print("─────────────────────────────────────────────────────────────")

    # ── Print access URL ─────────────────────────────────────────────────────
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
    except Exception:
        ip = "127.0.0.1"

    print(f"\n[OK ] Open in your browser:  http://{ip}:{args.port}")
    print("      Press Ctrl+C to stop.\n")

    # ── Inference runs in background thread ──────────────────────────────────
    t = threading.Thread(
        target=inference_loop,
        args=(cap, pfld, pose, phone, args.width),
        daemon=True
    )
    t.start()

    # ── Flask blocks here until Ctrl+C ───────────────────────────────────────
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)

    cap.release()
    print("\n[DONE] Test finished.")


if __name__ == "__main__":
    main()
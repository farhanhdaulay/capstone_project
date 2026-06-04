# src/dms/live_stream.py
# DMS Live Stream ??? view at http://<jetson-ip>:5000
# Run: pdm run python src/dms/live_stream.py
# Run USB cam: pdm run python src/dms/live_stream.py --source 0

import sys
import time
import threading
import argparse
import cv2
from flask import Flask, Response

sys.path.insert(0, 'src')
from dms.modules.camera    import CameraCapture
from dms.modules.pfld      import PFLDDetector
from dms.modules.head_pose import HeadPose6DRepNet
from dms.modules.phone     import PhoneDetector

parser = argparse.ArgumentParser()
parser.add_argument('--source',   default='csi')
parser.add_argument('--pfld',     default='models/pfld_106_lite.onnx')
parser.add_argument('--headpose', default='models/6drepnet360.onnx')
parser.add_argument('--phone',    default='models/yolo26n.pt')
parser.add_argument('--port',     type=int, default=5000)
parser.add_argument('--width',    type=int, default=640)
parser.add_argument('--height',   type=int, default=480)
args = parser.parse_args()

app = Flask(__name__)

latest_frame = None
frame_lock   = threading.Lock()

# Overlay 
def draw(frame, pfld_r, hp_r, phone_r, fps):
    h, w = frame.shape[:2]

    # Determine alert state 
    if pfld_r.get('drowsy') or phone_r.get('critical') or hp_r.get('distracted_crit'):
        state      = 'CRITICAL'
        state_col  = (0, 0, 220)
    elif pfld_r.get('ear_warn') or pfld_r.get('yawning') or \
         hp_r.get('distracted_warn') or pfld_r.get('no_face'):
        state      = 'WARNING'
        state_col  = (0, 140, 255)
    else:
        state      = 'NORMAL'
        state_col  = (0, 200, 80)

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 44), (15, 15, 15), -1)
    cv2.putText(frame, f'STATE: {state}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, state_col, 2)
    cv2.putText(frame, f'{fps:.1f} FPS', (w - 110, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 1)

    # Face bbox + landmarks
    if not pfld_r.get('no_face') and pfld_r.get('landmarks'):
        bbox = pfld_r.get('face_bbox')
        if bbox:
            x1, y1, x2, y2 = bbox
            lm_col = (0, 0, 220) if pfld_r.get('drowsy') else (0, 220, 80)
            cv2.rectangle(frame, (x1, y1), (x2, y2), lm_col, 2)
            for (lx, ly) in pfld_r['landmarks']:
                cv2.circle(frame, (int(x1 + lx), int(y1 + ly)), 1, lm_col, -1)

    # Bottom info panel 
    panel_y = h - 90
    cv2.rectangle(frame, (0, panel_y), (w, h), (15, 15, 15), -1)

    # EAR
    ear     = pfld_r.get('EAR') or 0.0
    ear_col = (0, 0, 220) if pfld_r.get('drowsy') else \
              (0, 140, 255) if pfld_r.get('ear_warn') else (0, 220, 80)
    cv2.putText(frame, f'EAR: {ear:.3f}', (10, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, ear_col, 2)

    # MAR
    mar     = pfld_r.get('MAR') or 0.0
    mar_col = (0, 140, 255) if pfld_r.get('yawning') else (0, 220, 80)
    cv2.putText(frame, f'MAR: {mar:.3f}', (10, panel_y + 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, mar_col, 2)

    # Drowsy / Yawn labels
    if pfld_r.get('drowsy'):
        cv2.putText(frame, 'DROWSY!', (160, panel_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 220), 2)
    elif pfld_r.get('ear_warn'):
        cv2.putText(frame, 'EYES CLOSING', (160, panel_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 140, 255), 2)

    if pfld_r.get('yawning'):
        cv2.putText(frame, 'YAWNING!', (160, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)

    # No face
    if pfld_r.get('no_face'):
        cv2.putText(frame, 'NO FACE', (160, panel_y + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 140, 255), 2)

    # Head pose
    pitch, yaw, roll = hp_r['pitch'], hp_r['yaw'], hp_r['roll']
    hp_col = (0, 0, 220) if hp_r.get('distracted_crit') else \
             (0, 140, 255) if hp_r.get('distracted_warn') else (180, 180, 180)
    cv2.putText(frame,
                f'P:{pitch:+.1f} Y:{yaw:+.1f} R:{roll:+.1f}',
                (w // 2 - 100, panel_y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.60, hp_col, 2)
    if hp_r.get('distracted_crit'):
        cv2.putText(frame, 'HEAD DISTRACTED!', (w // 2 - 100, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 0, 220), 2)
    elif hp_r.get('distracted_warn'):
        cv2.putText(frame, 'HEAD TURNING', (w // 2 - 100, panel_y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 140, 255), 2)

    # Phone
    if phone_r.get('phone_detected'):
        ph_col = (0, 0, 220) if phone_r.get('critical') else (0, 140, 255)
        cv2.putText(frame,
                    f"PHONE {phone_r['confidence']:.0%}",
                    (w - 160, panel_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, ph_col, 2)
        bbox = phone_r.get('bbox')
        if bbox:
            bx1, by1, bx2, by2 = bbox
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), ph_col, 2)
            cv2.putText(frame, 'phone', (bx1, by1 - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, ph_col, 1)

    return frame


# Inference thread 
def inference_loop():
    global latest_frame

    print("Loading models...")
    pfld  = PFLDDetector(model_path=args.pfld)
    hp    = HeadPose6DRepNet(model_path=args.headpose)
    phone = PhoneDetector(model_path=args.phone)
    cam   = CameraCapture(source=args.source,
                          width=args.width, height=args.height, flip=0)

    assert cam.is_opened(), \
        f"Camera failed to open (source={args.source})"

    print("Warming up models...")
    ret, wf = cam.read()
    if ret and wf is not None:
        pfld.process(wf)
        hp.estimate(wf)
        phone.detect(wf)

    print(f"\n???  Stream ready ??? open http://0.0.0.0:{args.port} in your browser\n")

    fps_t, fps_cnt, fps = time.time(), 0, 0.0

    while True:
        ret, frame = cam.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        pfld_r  = pfld.process(frame)
        hp_r    = hp.estimate(frame, pfld_r.get('face_bbox'))
        phone_r = phone.detect(frame)

        fps_cnt += 1
        elapsed  = time.time() - fps_t
        if elapsed >= 2.0:
            fps     = fps_cnt / elapsed
            fps_t   = time.time()
            fps_cnt = 0

        annotated = draw(frame, pfld_r, hp_r, phone_r, fps)

        with frame_lock:
            latest_frame = annotated


# MJPEG generator
def generate():
    while True:
        with frame_lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.033)
            continue
        _, buf = cv2.imencode('.jpg', frame,
                              [cv2.IMWRITE_JPEG_QUALITY, 80])
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'
               + buf.tobytes() + b'\r\n')
        time.sleep(0.033)


# Routes
@app.route('/')
def index():
    return '''<html><body style="margin:0;background:#000">
    <img src="/video" style="width:100%;height:100vh;object-fit:contain">
    </body></html>'''

@app.route('/video')
def video():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# Main 
if __name__ == '__main__':
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    # Give inference thread 2s to start up before Flask begins
    time.sleep(2)
    app.run(host='0.0.0.0', port=args.port, threaded=True)  # nosec B104 # intentional bind for Docker container

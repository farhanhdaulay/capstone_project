# src/dms/main.py
"""
main.py -- DMS master entry point.

CORRECT way to run (always from project root):
    cd ~/dms_project
    pdm run python -m dms.main

Options:
    --no-window     No cv2.imshow window (use over SSH)
    --no-stream     Disable browser MJPEG stream
    --port N        Stream port (default from config, 5000)
    --log-dir PATH  Session log directory
"""
from __future__ import annotations
from dms.modules.calibrator import EARCalibrator

import argparse
import logging
import os
import queue
import select
import signal
import socket
import sys
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from dms.modules.alert         import AlertController
from dms.modules.state_machine import DMSStateMachine, DMSEvent

# Allow `python src/dms/main.py` as a fallback
if __name__ == "__main__" and __package__ is None:
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

import dms.config as cfg

from dms.modules.camera import Camera


def _safe_import(module_path: str, class_name: str):
    """Import class_name from module_path, return None if unavailable."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError) as exc:
        logging.getLogger("dms.main").warning(
            "Could not import %s from %s: %s -- that feature will be disabled.",
            class_name, module_path, exc,
        )
        return None


_FaceDetector      = _safe_import("dms.modules.face_detector", "FaceDetector")
_PFLDLandmarker    = _safe_import("dms.modules.pfld",          "PFLDDetector")
_draw_lm           = _safe_import("dms.modules.pfld",          "draw_landmarks")
_HeadPoseEstimator = _safe_import("dms.modules.head_pose",     "HeadPoseEstimator")
_PhoneDetector     = _safe_import("dms.modules.phone",         "PhoneDetector")
_IMUReader         = _safe_import("dms.modules.imu",           "IMUReader")


# ---------------------------------------------------------------------------
# MJPEG stream
# ---------------------------------------------------------------------------
# Architecture:
#   main loop  -->  _frame_queue (raw BGR frames, maxsize=1, drop-oldest)
#   _encoder thread reads queue, encodes JPEG, writes to _latest_jpeg
#   _mjpeg_server thread reads _latest_jpeg and pushes to connected clients
#
# Keeping JPEG encoding off the main loop removes ~5-15ms per frame of
# latency on Jetson.  The queue maxsize=1 means the encoder always gets
# the newest frame and old frames are dropped, preventing build-up.
# ---------------------------------------------------------------------------

_frame_queue: queue.Queue = queue.Queue(maxsize=1)

_latest_jpeg: bytes = b""
_jpeg_lock          = threading.Lock()
_jpeg_event         = threading.Event()   # set whenever a new JPEG is ready


# A 1x1 black placeholder JPEG sent to clients before the first real frame.
_PLACEHOLDER_JPEG: bytes = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    b"\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t"
    b"\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a"
    b"\x1f\x1e\x1d\x1a\x1c\x1c $.' \",#\x1c\x1c(7),01444\x1f'9=82<.342\x1e"
    b"\x1e=49=\x1b\x1b\x1e' !&\x1e\x1b\x1c\x1c\xff\xc0\x00\x0b\x08"
    b"\x00\x01\x00\x01\x01\x01\x11\x00\xff\xc4\x00\x1f\x00\x00\x01"
    b"\x05\x01\x01\x01\x01\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00"
    b"\x01\x02\x03\x04\x05\x06\x07\x08\t\n\x0b\xff\xc4\x00\xb5\x10"
    b"\x00\x02\x01\x03\x03\x02\x04\x03\x05\x05\x04\x04\x00\x00\x01"
    b"\x7d\x01\x02\x03\x00\x04\x11\x05\x12!1A\x06\x13Qa\x07\"q\x142"
    b"\x81\x91\xa1\x08#B\xb1\xc1\x15R\xd1\xf0$3br\x82\t\n\x16\x17\x18"
    b"\x19\x1a%&'()*456789:CDEFGHIJSTUVWXYZcdefghijstuvwxyz\x83\x84\x85"
    b"\x86\x87\x88\x89\x8a\x92\x93\x94\x95\x96\x97\x98\x99\x9a\xa2\xa3"
    b"\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba"
    b"\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xd2\xd3\xd4\xd5\xd6\xd7\xd8"
    b"\xd9\xda\xe1\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xf1\xf2\xf3\xf4"
    b"\xf5\xf6\xf7\xf8\xf9\xfa\xff\xda\x00\x08\x01\x01\x00\x00?\x00"
    b"\xfb\xd4P\x00\x00\x00\x1f\xff\xd9"
)

# Minimal HTML -- NO JavaScript reconnect timer.
# The browser's built-in multipart/x-mixed-replace handling is reliable
# enough; adding a JS timer only causes the periodic black flash.
_HTML_PAGE = b"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <title>DMS Live</title>
  <style>
    * { box-sizing:border-box; margin:0; padding:0; }
    body { background:#0d0d0d; display:flex; flex-direction:column;
           align-items:center; min-height:100vh; font-family:monospace;
           color:#ccc; padding:16px; gap:12px; }
    h1   { font-size:.85rem; letter-spacing:.12em; color:#555;
           text-transform:uppercase; }
    img  { width:100%; max-width:860px; border:1px solid #222;
           border-radius:6px; image-rendering:auto; }
    p    { font-size:.7rem; color:#333; }
  </style>
</head>
<body>
  <h1>Driver Monitoring System &mdash; live feed</h1>
  <img src="/stream" alt="DMS camera feed"/>
  <p>Jetson Orin Nano Super &bull; MJPEG</p>
</body>
</html>
"""


def _get_local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _encoder_thread(quality: int) -> None:
    """
    Dedicated thread: pulls raw BGR frames from _frame_queue, encodes to JPEG,
    stores result in _latest_jpeg, and signals _jpeg_event.
    Runs independently of the MJPEG server so encoding never blocks pushing.
    """
    global _latest_jpeg
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    while True:
        frame = _frame_queue.get()       # blocks until a frame is available
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()
            _jpeg_event.set()


def _mjpeg_server(port: int) -> None:
    log = logging.getLogger("dms.stream")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))  # nosec B104 # intentional bind for Docker container
    srv.listen(16)
    srv.setblocking(False)

    stream_clients: list[socket.socket] = []

    def _accept() -> None:
        try:
            conn, addr = srv.accept()
        except BlockingIOError:
            return
        # Disable Nagle -- send each MJPEG chunk immediately without buffering
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.settimeout(2.0)
        try:
            raw = conn.recv(2048).decode(errors="ignore")
        except Exception:
            conn.close()
            return
        path = ""
        for line in raw.splitlines():
            if line.startswith("GET "):
                path = line.split()[1].split("?")[0]
                break

        if path == "/stream":
            hdr = (
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n"
                "Cache-Control: no-cache, no-store, must-revalidate\r\n"
                "Pragma: no-cache\r\n"
                "Expires: 0\r\n"
                "Connection: keep-alive\r\n\r\n"
            ).encode()
            try:
                conn.sendall(hdr)
                conn.setblocking(False)
                stream_clients.append(conn)
                log.info("Stream client connected: %s", addr)
            except Exception:
                conn.close()

        elif path in ("/", "/index.html"):
            resp = (
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                b"Content-Length: " + str(len(_HTML_PAGE)).encode() +
                b"\r\nConnection: close\r\n\r\n" + _HTML_PAGE
            )
            try:
                conn.sendall(resp)
            except Exception:
                pass
            conn.close()

        else:
            conn.sendall(b"HTTP/1.1 404 Not Found\r\nConnection: close\r\n\r\n")
            conn.close()

    def _push(jpeg: bytes) -> None:
        chunk = (
            b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
        )
        dead: list[socket.socket] = []
        for c in stream_clients:
            try:
                c.sendall(chunk)
            except Exception:
                dead.append(c)
        for c in dead:
            stream_clients.remove(c)
            try:
                c.close()
            except Exception:
                pass

    # Send placeholder once so the browser img tag shows something immediately
    # (will be replaced by the first real frame within a frame or two)
    _last_jpeg_sent: list[bytes] = [b""]

    while True:
        # Accept new connections (non-blocking)
        r, _, _ = select.select([srv], [], [], 0.002)
        if r:
            _accept()

        if not stream_clients:
            # No clients -- wait briefly then check for new connections
            time.sleep(0.01)
            continue

        # Wait for a new JPEG (up to 100ms so we keep accepting connections)
        got_new = _jpeg_event.wait(timeout=0.1)
        if got_new:
            _jpeg_event.clear()
            with _jpeg_lock:
                jpeg = _latest_jpeg
            if jpeg:
                _last_jpeg_sent[0] = jpeg
                _push(jpeg)
        # If no new frame arrived within timeout, do NOT push anything.
        # Pushing the same frame repeatedly causes the browser to flicker
        # because it re-renders the image on every multipart boundary.


def _enqueue_frame(frame: np.ndarray) -> None:
    """
    Put frame on the encode queue.  If the queue is full (encoder is busy),
    drop the oldest frame and put the new one -- always encode the latest.
    """
    try:
        _frame_queue.put_nowait(frame.copy())
    except queue.Full:
        try:
            _frame_queue.get_nowait()   # discard stale frame
        except queue.Empty:
            pass
        try:
            _frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str) -> None:
    os.makedirs(log_dir, exist_ok=True)
    stamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = os.path.join(log_dir, "session_{}.log".format(stamp))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s -- %(message)s",
        handlers=[
            logging.FileHandler(logfile),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logging.info("Log file: %s", logfile)


# ---------------------------------------------------------------------------
# OSD overlay
# ---------------------------------------------------------------------------

_STATE_COLORS = {
    "NORMAL":   (0, 200, 0),
    "WARNING":  (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}


def _draw_overlay(frame: np.ndarray, state_name: str,
                  event: DMSEvent, fps: float) -> np.ndarray:
    h, w  = frame.shape[:2]
    color = _STATE_COLORS.get(state_name, (255, 255, 255))
    cv2.rectangle(frame, (0, 0), (w, 36), (0, 0, 0), -1)
    cv2.putText(frame, "DMS: {}".format(state_name), (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, "FPS {:.1f}".format(fps), (w - 110, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 180, 180), 2)
    for i, txt in enumerate([
        "EAR  {:.3f}".format(event.ear),
        "MAR  {:.3f}".format(event.mar),
        "Yaw  {:+.1f}".format(event.yaw),
        "Ptch {:+.1f}".format(event.pitch),
        "Roll {:+.1f}".format(event.roll),
    ]):
        cv2.putText(frame, txt, (10, 68 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (210, 210, 210), 1)
    if event.active_labels:
        cv2.putText(frame, "  ".join(event.active_labels), (10, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2)
    return frame

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run(show_window: bool = True, stream: bool = True, port: int = 5000) -> None:
    logger = logging.getLogger(__name__)
    logger.info("=== DMS starting ===")

    if stream:
        ip = _get_local_ip()
        logger.info("Browser stream -- http://%s:%d", ip, port)
        print("\n  Open in browser:  http://{}:{}\n".format(ip, port))
        # Encoder thread: JPEG encoding off the main loop
        threading.Thread(
            target=_encoder_thread, args=(cfg.STREAM_JPEG_QUALITY,),
            daemon=True, name="jpeg-encoder"
        ).start()
        # Stream server thread
        threading.Thread(
            target=_mjpeg_server, args=(port,),
            daemon=True, name="mjpeg-server"
        ).start()

    # Camera
    camera = Camera(
        source=cfg.CAMERA_SOURCE,
        width=cfg.CAMERA_WIDTH,
        height=cfg.CAMERA_HEIGHT,
        fps=cfg.CAMERA_FPS,
        flip=cfg.CAMERA_FLIP,
    )
    camera.open()

    # FaceDetector
    face_det = None
    if _FaceDetector:
        try:
            face_det = _FaceDetector(
                dnn_proto      = getattr(cfg, "FACE_DNN_PROTO",  None),
                dnn_model      = getattr(cfg, "FACE_DNN_MODEL",  None),
                conf_threshold = cfg.FACE_CONF_THRESHOLD,
                scale_factor   = cfg.FACE_SCALE_FACTOR,
                min_neighbours = cfg.FACE_MIN_NEIGHBOURS,
                min_size       = cfg.FACE_MIN_SIZE,
                padding        = cfg.FACE_PADDING,
            )
        except Exception as exc:
            logger.warning("FaceDetector init failed: %s", exc)

    # PFLD
    pfld = None
    if _PFLDLandmarker and face_det:
        try:
            pfld = _PFLDLandmarker(model_path=cfg.PFLD_MODEL)
        except Exception as exc:
            logger.warning("PFLDLandmarker init failed: %s", exc)

    # Head pose
    head_pose = None
    if _HeadPoseEstimator and face_det:
        try:
            head_pose = _HeadPoseEstimator(model_path=cfg.HEAD_POSE_MODEL)
        except Exception as exc:
            logger.warning("HeadPoseEstimator init failed: %s", exc)

    # Phone detector
    phone_det = None
    if _PhoneDetector:
        try:
            phone_det = _PhoneDetector(
                engine_path = cfg.YOLO_ENGINE,
                pt_path     = cfg.YOLO_PT,
                conf        = cfg.PHONE_CONF_THRESHOLD,
                class_id    = cfg.PHONE_CLASS_ID,
                consec      = cfg.PHONE_CONSEC_FRAMES,
            )
        except Exception as exc:
            logger.warning("PhoneDetector init failed: %s", exc)

    # IMU
    imu = None
    if cfg.IMU_ENABLED and _IMUReader:
        try:
            imu = _IMUReader(bus_num=cfg.IMU_BUS, address=cfg.IMU_ADDRESS)
        except Exception as exc:
            logger.warning("IMUReader init failed: %s", exc)

    # Alert + state machine
    alert = AlertController(
        pin_green  = cfg.ALERT_GPIO_GREEN,
        pin_yellow = cfg.ALERT_GPIO_YELLOW,
        pin_red    = cfg.ALERT_GPIO_RED,
        pin_vib    = cfg.ALERT_GPIO_VIB,
        mock       = cfg.ALERT_MOCK,
    )
    sm = DMSStateMachine(
        alert             = alert,
        warning_duration  = cfg.WARNING_DURATION_S,
        critical_duration = cfg.CRITICAL_DURATION_S,
    )

    # Signal handling
    _running = True

    def _handle_exit(sig, frame):
        nonlocal _running
        _running = False

    signal.signal(signal.SIGINT,  _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    fps       = 0.0
    frame_cnt = 0
    fps_t0    = time.monotonic()
    logger.info("Loop running -- Ctrl-C or 'q' to stop.")
    
    # Calibrator
    cal           = EARCalibrator(
        duration_s   = cfg.EAR_CALIBRATION_SECS,
        drowsy_ratio = cfg.EAR_DROWSY_RATIO,
        fallback_ear = cfg.EAR_THRESHOLD,
    )
    ear_threshold = cfg.EAR_THRESHOLD   # live threshold; replaced once cal.done
    
    # Consecutive-frame counters (debounce) 
    _ear_consec   = 0
    _mar_consec   = 0
    _dist_consec  = 0
    _frame_idx    = 0
    
    # -- Background inference thread ------------------------------------------
    import queue as _queue
    
    _infer_input  = _queue.Queue(maxsize=1)   # (small_frame, small_bbox)
    _infer_output = _queue.Queue(maxsize=1)   # latest result dict
    
    _last_infer_result = {"ear": 0.0, "mar": 0.0, "landmarks": [],
                          "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                          "distracted": False}
                          
    def _inference_worker():
        _dist_consec_t = 0
        while True:
            small_f, small_b = _infer_input.get()
            out = dict(_last_infer_result)
            # PFLD
            if pfld and small_b is not None:
                try:
                    r = pfld.run(small_f, small_b)
                    if r:
                        out["ear"] = r.get("ear") or 0.0
                        out["mar"] = r.get("mar") or 0.0
                        out["landmarks"] = [
                            (lx / INFER_SCALE, ly / INFER_SCALE)
                            for lx, ly in r.get("landmarks", [])
                        ]
                except Exception:
                    pass
            # Head pose
            if head_pose and small_b is not None:
                try:
                    hp = head_pose.run(small_f, small_b)
                    if hp:
                        out["yaw"]   = hp.get("yaw",   0.0)
                        out["pitch"] = hp.get("pitch", 0.0)
                        out["roll"]  = hp.get("roll",  0.0)
                        if (abs(out["yaw"]) > cfg.YAW_THRESHOLD or
                                abs(out["pitch"]) > cfg.PITCH_THRESHOLD):
                            _dist_consec_t += 1
                        else:
                            _dist_consec_t = 0
                        out["distracted"] = _dist_consec_t >= cfg.DISTRACTION_FRAMES
                except Exception:
                    pass
            # Publish result (drop stale if consumer is slow)
            try:
                _infer_output.put_nowait(out)
            except _queue.Full:
                try: 
                  _infer_output.get_nowait()
                except _queue.Empty: 
                  pass
                _infer_output.put_nowait(out)

    threading.Thread(target=_inference_worker, daemon=True,
                 name="pfld-headpose").start()
                 
    try:
        while _running:
            # Capture
            frame = camera.read()
            if frame is None:
                logger.warning("No frame -- retrying...")
                time.sleep(0.05)
                continue

            event = DMSEvent()

            # Face detection - run on half-res frame for speed
            INFER_SCALE = 0.5
            small = cv2.resize(frame, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE)

            face_bbox  = None
            face_found = False
            if face_det:
                small_raw = face_det.detect(small)
                if small_raw is not None:
                    # Scale bbox back up to full-frame coords for drawing
                    sx1, sy1, sx2, sy2 = small_raw
                    face_bbox = (
                        int(sx1 / INFER_SCALE), int(sy1 / INFER_SCALE),
                        int(sx2 / INFER_SCALE), int(sy2 / INFER_SCALE),
                    )
                face_found = face_bbox is not None

            # Feed inference thread (non-blocking)
            if face_found and small_raw is not None:
                try:
                    _infer_input.put_nowait((small.copy(), small_raw))
                except _queue.Full:
                    pass

            # Consume latest result (use previous if not ready yet)
            try:
                _last_infer_result = _infer_output.get_nowait()
            except _queue.Empty:
                pass

            res = _last_infer_result
            event.ear        = res["ear"]
            event.mar        = res["mar"]
            event.yaw        = res["yaw"]
            event.pitch      = res["pitch"]
            event.roll       = res["roll"]
            event.distracted = res["distracted"]

            # Calibrator + drowsy/yawn debounce (stays on main thread)
            raw_ear = res["ear"]
            raw_mar = res["mar"]
            if raw_ear > 0.0:
                if not cal.done:
                    cal.update(raw_ear, raw_mar)
                    if cal.done:
                        ear_threshold = cal.ear_threshold
                        logger.info("Calibration complete -- EAR threshold %.4f", ear_threshold)
                if raw_ear < ear_threshold:
                    _ear_consec += 1
                else:
                    _ear_consec = 0
                event.drowsy = _ear_consec >= cfg.EAR_CONSEC_FRAMES

            if raw_mar > cfg.MAR_THRESHOLD:
                _mar_consec += 1
            else:
                _mar_consec = 0
            event.yawning = _mar_consec >= cfg.MAR_CONSEC_FRAMES

            if _draw_lm and res["landmarks"]:
                _draw_lm(frame, res["landmarks"])
            
            # Phone 
            if phone_det:
                try:
                    event.phone = phone_det.detect(frame)
                    if event.phone:
                        phone_det.draw(frame)
                except Exception as exc:
                    logger.debug("PhoneDetector error: %s", exc)

            # IMU
            if imu:
                try:
                    roll, _ = imu.read()
                    event.roll     = roll
                    event.imu_tilt = abs(roll) > cfg.IMU_TILT_THRESHOLD
                except Exception as exc:
                    logger.debug("IMU error: %s", exc)

            # State machine
            sm.update(event)

            # FPS counter (update every 30 frames)
            frame_cnt += 1
            if frame_cnt >= 30:
                fps       = frame_cnt / (time.monotonic() - fps_t0)
                fps_t0    = time.monotonic()
                frame_cnt = 0

            # OSD overlay
            # Show calibration progress on-screen
            if not cal.done:
                pct = int(cal.progress * 100)
                msg = f"CALIBRATING -- keep eyes open  {pct}%"
                cv2.putText(frame, msg, (10, frame.shape[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            _draw_overlay(frame, sm.state_name, event, fps)

            # Push to stream encoder (non-blocking, drops stale frames)
            if stream:
                _enqueue_frame(frame)

            # Local window
            if show_window:
                cv2.imshow("DMS", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    finally:
        logger.info("Shutting down...")
        sm.log_stats()
        alert.cleanup()
        camera.release()
        if imu:
            imu.close()
        if show_window:
            cv2.destroyAllWindows()
        logger.info("=== DMS stopped ===")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Driver Monitoring System")
    parser.add_argument("--no-window", action="store_true",
                        help="No cv2.imshow window (use over SSH).")
    parser.add_argument("--no-stream", action="store_true",
                        help="Disable MJPEG browser stream.")
    parser.add_argument("--port", type=int, default=cfg.STREAM_PORT,
                        help="Browser stream port (default {}).".format(cfg.STREAM_PORT))
    parser.add_argument("--log-dir", default=cfg.LOG_DIR,
                        help="Session log directory.")
    args = parser.parse_args()

    _setup_logging(args.log_dir)
    run(
        show_window = not args.no_window,
        stream      = not args.no_stream,
        port        = args.port,
    )


if __name__ == "__main__":
    main()
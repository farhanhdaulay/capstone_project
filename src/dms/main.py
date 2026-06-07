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
from typing import Any, Optional, Tuple, Dict

import cv2
import numpy as np

from dms.modules.calibrator import EARCalibrator
from dms.modules.alert         import AlertController
from dms.modules.state_machine import DMSStateMachine, DMSEvent
import dms.config as cfg
from dms.modules.camera import Camera
from dms.healthcheck import start_in_thread as start_healthz

# Allow `python src/dms/main.py` as a fallback
if __name__ == "__main__" and __package__ is None:
    _src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

def _safe_import(module_path: str, class_name: str) -> Any:
    """
    Safely imports a class from a module, bypassing fatal errors if dependencies are missing.

    Args:
        module_path (str): The dot-separated path to the Python module.
        class_name (str): The name of the class to import.

    Returns:
        Any: The imported class type, or None if the import fails.
    """
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

_frame_queue: queue.Queue = queue.Queue(maxsize=1)
_latest_jpeg: bytes = b""
_jpeg_lock          = threading.Lock()
_jpeg_event         = threading.Event()

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
    """
    Determines the local IP address of the machine for network streaming.

    Returns:
        str: The local IPv4 address, or '127.0.0.1' if disconnected.
    """
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
    Background worker thread that continually encodes raw OpenCV frames to JPEG format.

    Args:
        quality (int): JPEG encoding quality level (0-100).
    """
    global _latest_jpeg
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    while True:
        frame = _frame_queue.get()
        ok, buf = cv2.imencode(".jpg", frame, encode_params)
        if ok:
            with _jpeg_lock:
                _latest_jpeg = buf.tobytes()
            _jpeg_event.set()

def _mjpeg_server(port: int) -> None:
    """
    Runs a lightweight HTTP server to stream MJPEG payloads to connected browser clients.

    Args:
        port (int): The network port to bind the stream to.
    """
    log = logging.getLogger("dms.stream")
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))  # nosec B104
    srv.listen(16)
    srv.setblocking(False)

    stream_clients: list[socket.socket] = []

    def _accept() -> None:
        try:
            conn, addr = srv.accept()
        except BlockingIOError:
            return
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

    _last_jpeg_sent: list[bytes] = [b""]
    while True:
        r, _, _ = select.select([srv], [], [], 0.002)
        if r:
            _accept()
        if not stream_clients:
            time.sleep(0.01)
            continue
        got_new = _jpeg_event.wait(timeout=0.1)
        if got_new:
            _jpeg_event.clear()
            with _jpeg_lock:
                jpeg = _latest_jpeg
            if jpeg:
                _last_jpeg_sent[0] = jpeg
                _push(jpeg)

def _enqueue_frame(frame: np.ndarray) -> None:
    """
    Pushes a new frame to the encoding queue, discarding the oldest if the queue is full.

    Args:
        frame (np.ndarray): The raw BGR frame captured from the camera.
    """
    try:
        _frame_queue.put_nowait(frame.copy())
    except queue.Full:
        try:
            _frame_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            _frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

# ---------------------------------------------------------------------------
# Logging & OSD
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: str) -> None:
    """
    Configures standard Python logging to output both to the terminal and a timestamped file.

    Args:
        log_dir (str): The directory path to store the session log files.
    """
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

_STATE_COLORS = {
    "NORMAL":   (0, 200, 0),
    "WARNING":  (0, 165, 255),
    "CRITICAL": (0, 0, 255),
}

def _draw_overlay(frame: np.ndarray, state_name: str,
                  event: DMSEvent, fps: float) -> np.ndarray:
    """
    Draws the On-Screen Display (OSD) overlay containing telemetry and active alerts onto the video frame.

    Args:
        frame (np.ndarray): The BGR image frame to be drawn on.
        state_name (str): The current state of the DMS state machine (e.g., 'NORMAL', 'CRITICAL').
        event (DMSEvent): The current state event containing raw metrics like EAR, MAR, and pose angles.
        fps (float): The current calculated frames per second of the pipeline.

    Returns:
        np.ndarray: The modified image frame with the OSD applied.
    """
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
    """
    Initializes hardware peripherals, starts background GPU inference workers, 
    and executes the primary Driver Monitoring System asynchronous loop.

    This function sets up a Producer-Consumer architecture where the main thread
    handles physical camera I/O and state machine updates, while heavy YOLO and 
    ONNX math is dispatched to dedicated background threads to prevent GPU starvation.

    Args:
        show_window (bool): If True, displays a local OpenCV GUI window. Defaults to True.
        stream (bool): If True, spins up a background thread to host an MJPEG web server. Defaults to True.
        port (int): The network port used for the MJPEG stream. Defaults to 5000.
    """
    logger = logging.getLogger(__name__)
    logger.info("=== DMS starting ===")

    if stream:
        ip = _get_local_ip()
        logger.info("Browser stream -- http://%s:%d", ip, port)
        print("\n  Open in browser:  http://{}:{}\n".format(ip, port))
        threading.Thread(
            target=_encoder_thread, args=(cfg.STREAM_JPEG_QUALITY,),
            daemon=True, name="jpeg-encoder"
        ).start()
        threading.Thread(
            target=_mjpeg_server, args=(port,),
            daemon=True, name="mjpeg-server"
        ).start()

    camera = Camera(
        source=0,
        width=cfg.CAMERA_WIDTH,
        height=cfg.CAMERA_HEIGHT,
        fps=cfg.CAMERA_FPS,
        flip=cfg.CAMERA_FLIP,
    )
    camera.open()

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

    pfld = None
    if _PFLDLandmarker and face_det:
        try:
            pfld = _PFLDLandmarker(model_path=cfg.PFLD_MODEL)
        except Exception as exc:
            logger.warning("PFLDLandmarker init failed: %s", exc)

    head_pose = None
    if _HeadPoseEstimator and face_det:
        try:
            head_pose = _HeadPoseEstimator(model_path=cfg.HEAD_POSE_MODEL)
        except Exception as exc:
            logger.warning("HeadPoseEstimator init failed: %s", exc)

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

    imu = None
    if cfg.IMU_ENABLED and _IMUReader:
        try:
            imu = _IMUReader(bus_num=cfg.IMU_BUS, address=cfg.IMU_ADDRESS)
        except Exception as exc:
            logger.warning("IMUReader init failed: %s", exc)

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

    _running = True
    def _handle_exit(sig: int, frame: Any) -> None:
        nonlocal _running
        _running = False

    signal.signal(signal.SIGINT,  _handle_exit)
    signal.signal(signal.SIGTERM, _handle_exit)

    fps       = 0.0
    frame_cnt = 0
    fps_t0    = time.monotonic()
    frame_latencies_ms = [] # a list to store the latency of every frame
    logger.info("Loop running -- Ctrl-C or 'q' to stop.")
    
    cal           = EARCalibrator(
        duration_s   = cfg.EAR_CALIBRATION_SECS,
        drowsy_ratio = cfg.EAR_DROWSY_RATIO,
        fallback_ear = cfg.EAR_THRESHOLD,
    )
    ear_threshold = cfg.EAR_THRESHOLD
    
    _ear_consec   = 0
    _mar_consec   = 0
    _frame_idx    = 0
    
    # -- Background inference thread ------------------------------------------
    import queue as _queue
    
    _infer_input: _queue.Queue[Tuple[np.ndarray, np.ndarray]] = _queue.Queue(maxsize=1)
    _infer_output: _queue.Queue[Tuple[Dict[str, Any], Optional[Tuple[int, int, int, int]]]] = _queue.Queue(maxsize=1)
    
    _last_infer_result = {"ear": 0.0, "mar": 0.0, "landmarks": [],
                          "yaw": 0.0, "pitch": 0.0, "roll": 0.0,
                          "distracted": False, "phone": False}
    _last_bbox = None
                          
    def _inference_worker() -> None:
        """
        Dedicated GPU thread that consumes frames from the input queue, processes 
        heavy AI workloads (YOLO, PFLD, HeadPose), and publishes telemetry to the output queue.
        """
        _dist_consec_t = 0
        INFER_SCALE = 0.5
        _infer_frame_count = 0
        _last_bbox = None
        while True:
            try:
                frame_full, small_f = _infer_input.get()
                _infer_frame_count += 1
            except Exception:
                continue

            out = dict(_last_infer_result)
            small_b = None

            # 1. Face YOLO
            # Frame Skip: Only run YOLO every 3rd frame (roughly 10 times a second)
            if _infer_frame_count % 3 == 0:
                if face_det:
                    try:
                        small_b = face_det.detect(small_f)
                        if small_b is not None:
                            _last_bbox = small_b  # Update only if a face is found
                    except Exception:
                        pass # Silently fail and reuse the old box
            # Reuse the last known box for skipped frames or failed detections
            small_b = _last_bbox
            if small_b is None:
                continue
            # 2. PFLD & Headpose
            if small_b is not None:
                if pfld:
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
                
                if head_pose:
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

            # 3. Phone YOLO
            out["phone"] = False
            if phone_det:
                try:
                    out["phone"] = phone_det.detect(frame_full)
                except Exception:
                    pass

            # 4. Publish result back to main thread
            try:
                _infer_output.put_nowait((out, small_b))
            except _queue.Full:
                try: 
                  _infer_output.get_nowait()
                except _queue.Empty: 
                  pass
                try:
                  _infer_output.put_nowait((out, small_b))
                except _queue.Full:
                  pass

    threading.Thread(target=_inference_worker, daemon=True,
                 name="gpu-inference-worker").start()
                 
    try:
        while _running:
            # 1. Hardware I/O (Main loop only blocks on the physical camera)
            loop_start = time.perf_counter() # START TIMER
            frame = camera.read()
            if frame is None:
                logger.warning("No frame -- retrying...")
                time.sleep(0.05)
                continue

            event = DMSEvent()
            INFER_SCALE = 0.5
            small = cv2.resize(frame, (0, 0), fx=INFER_SCALE, fy=INFER_SCALE)

            # 2. Feed the GPU worker immediately
            try:
                _infer_input.put_nowait((frame.copy(), small.copy()))
            except _queue.Full:
                pass

            # 3. Check for fresh GPU calculations (do not block)
            try:
                _last_infer_result, _last_bbox = _infer_output.get_nowait()
            except _queue.Empty:
                pass

            res = _last_infer_result
            event.ear        = res["ear"]
            event.mar        = res["mar"]
            event.yaw        = res["yaw"]
            event.pitch      = res["pitch"]
            event.roll       = res["roll"]
            event.distracted = res["distracted"]
            event.phone      = res.get("phone", False)

            # 4. CPU Light Logic (State Machine & Calibrator)
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

            # 5. Drawing & OSD
            if _draw_lm and res.get("landmarks"):
                _draw_lm(frame, res["landmarks"])
            
            # Since phone detector modifies its own state internally, 
            # we draw the last detected box if the event is active
            if phone_det and event.phone:
                try:
                    phone_det.draw(frame)
                except Exception:
                    pass

            if imu:
                try:
                    roll, _ = imu.read()
                    event.roll     = roll
                    event.imu_tilt = abs(roll) > cfg.IMU_TILT_THRESHOLD
                except Exception as exc:
                    logger.debug("IMU error: %s", exc)

            sm.update(event)

            frame_cnt += 1
            if frame_cnt >= 30:
                fps       = frame_cnt / (time.monotonic() - fps_t0)
                fps_t0    = time.monotonic()
                frame_cnt = 0

            if not cal.done:
                pct = int(cal.progress * 100)
                msg = f"CALIBRATING -- keep eyes open  {pct}%"
                cv2.putText(frame, msg, (10, frame.shape[0] - 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            _draw_overlay(frame, sm.state_name, event, fps)
            
            loop_end = time.perf_counter()
            frame_latencies_ms.append((loop_end - loop_start) * 1000)
            
            # 6. Stream and Display
            if stream:
                _enqueue_frame(frame)

            if show_window:
                cv2.imshow("DMS", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break

    finally:
        logger.info("Shutting down...")
        # Calculate p50, p95, p99
        if frame_latencies_ms:
            # We skip the first 30 frames to ignore the heavy startup/initialization time
            warm_latencies = frame_latencies_ms[30:] if len(frame_latencies_ms) > 30 else frame_latencies_ms
            
            p50 = np.percentile(warm_latencies, 50)
            p95 = np.percentile(warm_latencies, 95)
            p99 = np.percentile(warm_latencies, 99)
            
            # store the letency report inside session logs with logger.info
            logger.info("="*40)
            logger.info("LATENCY REPORT (ms)")
            logger.info("="*40)
            logger.info(f"p50 (Median) : {p50:.1f} ms")
            logger.info(f"p95          : {p95:.1f} ms")
            logger.info(f"p99 (Tail)   : {p99:.1f} ms")
            logger.info("="*40)

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
    """
    Command-line interface entry point for the Driver Monitoring System.
    
    Parses execution arguments and initiates the main application loop.
    """
    start_healthz()
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
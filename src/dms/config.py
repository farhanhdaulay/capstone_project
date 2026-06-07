# src/dms/config.py
"""
Central configuration -- all thresholds, paths, and hardware settings.
Edit this file to tune the DMS without touching any other module.

Jetson Orin Nano Super -- GPIO.BOARD mode (physical pin numbers).
Pinout reference (Figure 3-1):
  Pin  1 = 3.3V         Pin  2 = 5V
  Pin  3 = I2C1_SDA     Pin  4 = 5V
  Pin  5 = I2C1_SCL     Pin  6 = GND
  Pin  9 = GND          Pin 14 = GND
  Pin 7 = GPIO12  -> green  LED
  Pin 29 = GPIO01  -> yellow LED
  Pin 31 = GPIO11  -> red    LED
  Pin 33 = GPIO13  -> vibration motor (via 2N7000 MOSFET)
"""
import os

# ---------------------------------------------------------------------------
# Project paths
# config.py lives at:  <project_root>/src/dms/config.py
#   dirname once  -> src/dms/
#   dirname twice -> src/
#   dirname three -> <project_root>/          <-- correct
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
LOG_DIR   = os.path.join(PROJECT_ROOT, "logs")

# ---------------------------------------------------------------------------
# Model files
# ---------------------------------------------------------------------------
PFLD_MODEL      = os.path.join(MODEL_DIR, "pfld_106_lite.onnx")
HEAD_POSE_MODEL = os.path.join(MODEL_DIR, "6drepnet360.onnx")
YOLO_ENGINE     = os.path.join(MODEL_DIR, "yolo26n.engine")   # TensorRT engine if built
YOLO_PT         = os.path.join(MODEL_DIR, "yolo26n.pt")     # fallback PyTorch weights

# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------
CAMERA_SOURCE = "/dev/video0"          # "csi" for IMX219, 0 for USB
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 320
CAMERA_FPS    = 30
CAMERA_FLIP   = 0              # nvvidconv flip-method (0=none, 2=180 deg)

# ---------------------------------------------------------------------------
# PFLD landmark indices
# pfld_106_lite.onnx actually outputs 136 values = 68 landmarks (iBUG 300-W)
# pfld.py auto-detects and uses the correct set -- these are kept for reference.
# ---------------------------------------------------------------------------
LEFT_EYE_IDX  = [36, 37, 38, 39, 40, 41]   # 68-pt iBUG outer->inner, upper+lower
RIGHT_EYE_IDX = [42, 43, 44, 45, 46, 47]
MOUTH_TOP_IDX = 51
MOUTH_BOT_IDX = 57
MOUTH_L_IDX   = 48
MOUTH_R_IDX   = 54
NUM_LANDMARKS = 68   # matches actual model output (136 / 2)

# ---------------------------------------------------------------------------
# EAR / MAR thresholds
# ---------------------------------------------------------------------------
EAR_THRESHOLD        = 0.20   # below -> eye closing / drowsy
EAR_CONSEC_FRAMES    = 3      # frames EAR must stay below threshold
MAR_THRESHOLD        = 0.65   # above -> yawning
MAR_CONSEC_FRAMES    = 2
EAR_CALIBRATION_SECS = 10.0   # seconds to collect baseline EAR (eyes open)
EAR_DROWSY_RATIO     = 0.75   # threshold = baseline_ear * this ratio

# ---------------------------------------------------------------------------
# Head pose thresholds (degrees)
# ---------------------------------------------------------------------------
YAW_THRESHOLD      = 30.0
PITCH_THRESHOLD    = 20.0
DISTRACTION_FRAMES = 4

# ---------------------------------------------------------------------------
# Phone detection
# ---------------------------------------------------------------------------
PHONE_CONF_THRESHOLD = 0.40
PHONE_CLASS_ID       = 67    # COCO class index for 'cell phone'
PHONE_CONSEC_FRAMES  = 2

# ---------------------------------------------------------------------------
# State machine timing
# ---------------------------------------------------------------------------
WARNING_DURATION_S  = 2.0
CRITICAL_DURATION_S = 3.0

# ---------------------------------------------------------------------------
# Alert / GPIO  (BOARD mode = physical pin numbers on 40-pin header)
# ---------------------------------------------------------------------------
ALERT_GPIO_GREEN  = 7     # GPIO12 -> green  LED (NORMAL)
ALERT_GPIO_YELLOW = 31     # GPIO01 -> yellow LED (WARNING)
ALERT_GPIO_RED    = 29     # GPIO11 -> red    LED (CRITICAL)
ALERT_GPIO_VIB    = 33     # GPIO13 -> vibration motor via 2N7000
ALERT_MOCK        = False  # True = print only, no real GPIO

# ---------------------------------------------------------------------------
# IMU (MPU6050)
# ---------------------------------------------------------------------------
IMU_BUS            = 1       # I2C bus 1 (SDA=Pin3, SCL=Pin5)
IMU_ADDRESS        = 0x68    # default MPU6050 I2C address
IMU_ENABLED        = True
IMU_TILT_THRESHOLD = 45.0    # degrees roll before flagging imu_tilt

# ---------------------------------------------------------------------------
# Browser MJPEG stream
# ---------------------------------------------------------------------------
STREAM_PORT         = 5000
STREAM_JPEG_QUALITY = 60

# ---------------------------------------------------------------------------
# Face detector
# ---------------------------------------------------------------------------
FACE_CONF_THRESHOLD = 0.6
FACE_SCALE_FACTOR   = 1.1      # Haar cascade scaleFactor
FACE_MIN_NEIGHBOURS = 5        # Haar cascade minNeighbors
FACE_MIN_SIZE       = (60, 60)
FACE_PADDING        = 0.20     # fractional padding around detected bbox
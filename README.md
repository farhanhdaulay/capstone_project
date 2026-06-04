# Capstone Project: Driver Monitoring System (Smart Car AI)

**Team members:**
1. Farhan Hikmatullah Daulay - 611451002
2. Kishore Sridhar - 611451003

---

Real-time drowsiness, yawning, and distraction detection running on a
**Jetson Orin Nano Super** with an **IMX219 CSI camera**.

---

## Hardware

| Component | Part | Interface |
|---|---|---|
| SBC | Jetson Orin Nano Super | – |
| Camera | IMX219 (Raspberry Pi v2) | CSI (GStreamer) |
| IMU | MPU6050 | I2C bus 1, addr 0x68 |
| LED Green | – | GPIO BCM 12 |
| LED Yellow | – | GPIO BCM 16 |
| LED Red | – | GPIO BCM 18 |
| Vibration Motor | via NPN/MOSFET | GPIO BCM 23 |
| USB Sound Card | – | ALSA / sounddevice |

---

## Detection Pipeline

    IMX219 --> FaceDetector --> PFLD ----------> EAR (drowsy) / MAR (yawn)
                            +-> 6DRepNet ------> Yaw / Pitch (distracted)
               Full frame  ---> YOLOv11n ------> Phone present
               MPU6050      -------------------> Roll / Pitch (IMU tilt)
                                                            |
                                                       StateMachine
                                                            |
                                                +-----------+-----------+
                                             NORMAL      WARNING    CRITICAL
                                            (Green)     (Yellow)    (Red)
                                                       +vib pulse  +vib cont
                                                                   +buzzer

---

## Project Layout

    dms_project/
    +-- src/dms/
    |   +-- main.py           Master loop
    |   +-- config.py         All thresholds & pin numbers  <- edit here
    |   +-- modules/
    |       +-- camera.py     IMX219 / USB capture
    |       +-- face_detector.py  ONNX face detector + Haar fallback
    |       +-- pfld.py         PFLD-106-lite -> EAR / MAR
    |       +-- head_pose.py    6DRepNet360 -> yaw / pitch
    |       +-- phone.py        YOLOv11n TensorRT -> phone detection
    |       +-- imu.py          MPU6050 I2C + Kalman filter
    |       +-- alert.py        LED / motor / buzzer controller
    |       +-- state_machine.py  NORMAL / WARNING / CRITICAL FSM
    +-- models/
    |   +-- pfld_106_lite.onnx
    |   +-- 6drepnet360.onnx
    |   +-- yolo11n.engine      (TensorRT - see Step 5 below)
    |   +-- yolo11n.pt
    +-- tests/
        +-- test_camera.py
        +-- test_imu.py
        +-- test_pfld.py
        +-- test_headpose.py
        +-- test_phone.py
        +-- test_alert.py
        +-- test_dms_all.py

---

## Setup

### 1 – Clone & initialise PDM

    git clone <your-repo>
    cd dms_project
    pip install pdm --break-system-packages
    pdm install

### 2 – I2C: enable & verify MPU6050

    # Enable I2C on Jetson (if not already)
    sudo usermod -aG i2c $USER
    sudo systemctl restart nvargus-daemon
    
    # Scan the bus
    i2cdetect -y -r 1
    # You should see 0x68 in the table

### 3 – GPIO permissions

    sudo groupadd -f gpio
    sudo usermod -aG gpio $USER
    sudo chmod a+rw /sys/class/gpio/export
    # OR add Jetson GPIO udev rules (see Jetson.GPIO README)

### 4 – USB Sound card

    # Check ALSA sees the USB card
    aplay -l
    # Should list a USB Audio Device
    
    # Test playback
    aplay /usr/share/sounds/alsa/Front_Center.wav
    
    # For sounddevice to auto-select the USB card set default:
    # /etc/asound.conf
    #   defaults.pcm.card 1
    #   defaults.ctl.card 1

### 5 – Build TensorRT engine for YOLOv11n

    # Export to engine (run once on the Jetson)
    pdm run python - <<'EOF_TRT'
    from ultralytics import YOLO
    model = YOLO("models/yolo11n.pt")
    model.export(format="engine", device=0, half=True,
                 imgsz=640, workspace=4)
    # Produces models/yolo11n.engine
    EOF_TRT

---

## Configuration (`src/dms/config.py`)

Key values to tune for your driver:

| Constant | Default | Meaning |
|---|---|---|
| `EAR_THRESHOLD` | 0.20 | Eye Aspect Ratio -> drowsy below this |
| `EAR_CONSEC_FRAMES` | 15 | Frames EAR must stay low |
| `MAR_THRESHOLD` | 0.55 | Mouth Aspect Ratio -> yawn above this |
| `YAW_THRESHOLD` | 30.0 | Head turn left/right limit |
| `PITCH_THRESHOLD` | 20.0 | Head up/down limit |
| `WARNING_DURATION_S` | 2.0 | Seconds in WARNING -> CRITICAL |
| `IMU_ENABLED` | False | Set `True` when MPU6050 is wired |
| `ALERT_MOCK` | True | Set `False` for real GPIO |

---

## Running

    # Full DMS with display window
    pdm run python -m dms.main
    
    # Headless (SSH session, no monitor)
    pdm run python -m dms.main --no-window
    
    # Override log directory
    pdm run python -m dms.main --log-dir /mnt/usb/logs

Press **q** or **ESC** in the window to stop. **Ctrl-C** works headless.

---

## Tests

Run individual module tests first before the full integration test:

    # Camera
    pdm run python tests/test_camera.py
    
    # IMU (SIMULATION if smbus2/hardware absent)
    pdm run python tests/test_imu.py --duration 10
    
    # PFLD landmarks
    pdm run python tests/test_pfld.py
    
    # Head pose
    pdm run python tests/test_headpose.py
    
    # Phone detection
    pdm run python tests/test_phone.py
    
    # Alert actuators (mock - safe without wiring)
    pdm run python tests/test_alert.py
    
    # Alert actuators (REAL GPIO - wire hardware first!)
    pdm run python tests/test_alert.py --real --hold 3
    
    # Full pipeline integration
    pdm run python tests/test_dms_all.py

---

## Alert State Machine

                            any_event sustained >= WARNING_DURATION_S
    NORMAL --------------------------------------------------> CRITICAL
      ^        any_event appears                                   |
      |   NORMAL ----------> WARNING ----------------------------> |
      |                        |  event clears                     |
      +------------------------+                                   |
      +----------------------------- event clears -----------------+

| State | Green LED | Yellow LED | Red LED | Vibration | Sound |
|---|:---:|:---:|:---:|:---:|:---:|
| NORMAL | ON | off | off | off | silent |
| WARNING | off | ON | off | short pulse | silent |
| CRITICAL | off | off | ON | continuous | alarm loop |

---

## Wiring Diagram

    Jetson 40-pin Header
    --------------------
    Pin  1 (3.3 V) ----------- MPU6050 VCC
    Pin  3 (I2C1 SDA) ------- MPU6050 SDA
    Pin  5 (I2C1 SCL) ------- MPU6050 SCL
    Pin  6 (GND) ------------ MPU6050 GND
    
    Pin 32 (BCM 12) --[330 O]-- Green  LED anode  -> GND
    Pin 36 (BCM 16) --[330 O]-- Yellow LED anode  -> GND
    Pin 12 (BCM 18) --[330 O]-- Red    LED anode  -> GND
    Pin 16 (BCM 23) --[base]--- NPN transistor -> vibration motor -> 5 V
                                 (collector -> motor +, emitter -> GND)
    
    USB Sound Card ------------ USB port -> Speaker / Buzzer

> **Note** – Jetson BCM numbering may differ from Raspberry Pi.
> Verify actual physical pin locations with `pinmux` or the Jetson
> Orin Nano pinout diagram before wiring.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `smbus2` import error | `pip install smbus2 --break-system-packages` |
| IMU reads all zeros | Check `i2cdetect -y -r 1`, verify SDA/SCL wiring |
| GPIO permission denied | `sudo usermod -aG gpio $USER` then re-login |
| No USB audio | `aplay -l` -> set `defaults.pcm.card` in `/etc/asound.conf` |
| TensorRT engine fails | Rebuild with `model.export(format="engine")` on Jetson |
| Low FPS (<15) | Reduce `CAMERA_WIDTH/HEIGHT`, enable TRT engine, use `--no-window` |
| Face not detected | Improve lighting; lower `FACE_CONF_THRESHOLD` in config |
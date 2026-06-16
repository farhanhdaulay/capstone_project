# Edge-AI Driver Monitoring System for Real-Time Drowsiness, Yawning, and Distraction Detection Using PFLD, 6DRepNet, and Yolov26n on NVIDIA Jetson Orin Nano Super by Farhan Hikmatullah Daulay and Kishore Sridhar at Tatung University

## Project Overview

The **Smart Car AI Driver Monitoring System (DMS)** is a real-time, edge-deployed safety pipeline engineered for the NVIDIA Jetson Orin Nano. Built for low-latency industrial and automotive applications, the system monitors both driver state and vehicle dynamics to proactively prevent accidents.

**Core Capabilities:**

- **Cognitive State Tracking:** Computes Eye Aspect Ratio (EAR) and Mouth Aspect Ratio (MAR) using PFLD-106 facial landmarks to detect micro-sleeps and yawning.
- **Attention Monitoring:** Utilizes 6DRepNet360 for high-precision head pose estimation (yaw/pitch) to track driver gaze and visual distractions.
- **Contraband Detection:** Deploys YOLOv26n model, accelerated via TensorRT, to detect cell phone usage at the wheel.
- **Direct-Edge Actuation:** Bypasses network latency by using a localized Finite State Machine (FSM) to trigger physical GPIO hardware (LEDs, vibration motors) and ALSA audio alerts instantly.

**Team members:**

1. Farhan Hikmatullah Daulay - 611451002
2. Kishore Sridhar - 611451003

---

Real-time drowsiness, yawning, and distraction detection running on a
**Jetson Orin Nano Super** with an **USB Camera**.

---

## Hardware

| Component       | Part                   | Interface          |
| --------------- | ---------------------- | ------------------ |
| SBC             | Jetson Orin Nano Super | –                  |
| Camera          | USB Camera             |
| LED Green       | –                      | GPIO pin 7         |
| LED Yellow      | –                      | GPIO pin 31        |
| LED Red         | –                      | GPIO pin 29        |
| Vibration Motor | via NPN/MOSFET         | GPIO pin 33        |
| USB Sound Card  | –                      | ALSA / sounddevice |

---

## Detection Pipeline

    USB Camera --> FaceDetector --> PFLD ----------> EAR (drowsy) / MAR (yawn)
                            +-> 6DRepNet ------> Yaw / Pitch (distracted)
               Full frame  ---> YOLOv26n ------> Phone present
                                                            |
                                                       StateMachine
                                                            |
                                                +-----------+-----------+
                                             NORMAL      WARNING    CRITICAL
                                            (Green)     (Yellow)    (Red)
                                                       +vib pulse  +vib cont
                                                                   +buzzer

---

## Telemetry & Communications (Architecture Note)

_Note: This project utilizes a direct-edge-actuation architecture rather than a distributed MQTT broker model._

- **Actuation:** Alerts are processed strictly on-device, directly triggering local Jetson GPIO pins (LEDs, Vibration) and the ALSA audio driver with zero network latency.
- **Telemetry Stream:** Live telemetry (EAR, MAR, Head Pose angles) is overlaid directly onto the video feed.
- **Network Video:** The system serves an MJPEG live feed and health-check endpoint via local HTTP (Ports 5000 and 8000).

## Project Layout

    dms_project/
    +-- .github/workflows
    |   +-- ci.yml
    |   +-- deploy.yml
    +-- deploy/
    |   +-- deploy.sh
    |   +-- docker-compose.yml
    |   +-- healthcheck.sh
    |   +-- power_profile.json
    |   +-- rollback.sh
    +-- models/
    |   +-- 6drepnet360.engine
    |   +-- pfld_106_lite.engine
    |   +-- yolo26n.engine
    +-- scripts/
    |   +-- parse_tegrastats.py
    |   +-- utilization.csv
    +-- src/dms/
    |   +-- main.py           Master loop
    |   +-- live_stream.py
    |   +-- healthcheck.py
    |   +-- config.py         All thresholds & pin numbers  <- edit here
    |   +-- modules/
    |       +-- alert.py          LED / motor / buzzer controller
    |       +-- calibrator.py     initial 10s calibration
    |       +-- camera.py         USB capture
    |       +-- face_detector.py  ONNX face detector + Haar fallback
    |       +-- head_pose.py      6DRepNet360 -> yaw / pitch
    |       +-- pfld.py           PFLD-106-lite -> EAR / MAR
    |       +-- phone.py          YOLOv26n TensorRT -> phone detection
    |       +-- state_machine.py  NORMAL / WARNING / CRITICAL FSM
    |       +-- trt_backend.py    having tensorRT engine to load the models
    +-- tests/
    |   +-- integration/
    |       +-- sample_frame.jpg
    |       +-- test_jetson_e2e.py
    |   +-- test_alert.py
    |   +-- test_calibrator.py
    |   +-- test_camera.py
    |   +-- test_config.py
    |   +-- test_face_detector.py
    |   +-- test_head_pose.py
    |   +-- test_healthcheck.py
    |   +-- test_pfld.py
    |   +-- test_phone.py
    |   +-- test_state_machine.py
    +-- Dockerfile
    +-- README.md
    +-- pyproject.toml
    +-- requirements.txt
    +-- test_video.mp4

---

## How to Demo (Simulation Mode)

If downloading this project on a machine without the custom Jetson GPIO pins, or a USB camera connected, we can use this command to run a full demonstration.

It safely bypasses the hardware requirements and runs the full TensorRT AI pipeline on a pre-recorded sample video.

**1. Clone the repository and navigate into it:**

```bash
git clone [https://github.com/farhanhdaulay/capstone_project.git](https://github.com/farhanhdaulay/capstone_project.git)
cd capstone_project
```

**2. Download the sample video if not able to pulled from github:**

```bash
wget -O test_video.mp4 https://raw.githubusercontent.com/farhanhdaulay/capstone_project/main/test_video.mp4
```

```bash
ls -lh test_video.mp4
```

**3. Push the latest image:**

```bash
docker pull ghcr.io/farhanhdaulay/capstone_project:latest
```

**4. Run the simulation:**

```bash
docker run -d \
  --name dms-core-sim \
  --runtime nvidia \
  --network host \
  --env LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/tegra \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/test_video.mp4:/app/test_video.mp4 \
  -e HEALTHZ_PORT=8000 \
  ghcr.io/farhanhdaulay/capstone_project:latest
```

**5. Check the logs:(Run this command until the test video finishes)**

```bash
docker logs dms-core-sim
```

**6. Check the live video:(Optional - only run if all the hardwares are connected):**

```bash
docker rm -f dms-core-sim
```

```bash
docker run -d \
  --name dms-core-live \
  --runtime nvidia \
  --network host \
  --privileged \
  --device /dev/video0:/dev/video0 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -e CAMERA_SOURCE=/dev/video0 \
  ghcr.io/farhanhdaulay/capstone_project:latest
```

## Setup without Docker

### 1 – Clone & initialise PDM

    git clone https://github.com/farhanhdaulay/capstone_project.git
    cd dms_project
    pip install pdm --break-system-packages
    pdm install

**2. Download the sample video if not able to pulled from github:**

```bash
wget -O test_video.mp4 https://raw.githubusercontent.com/farhanhdaulay/capstone_project/main/test_video.mp4
```

```bash
ls -lh test_video.mp4
```

### 3 – GPIO permissions

    sudo groupadd -f gpio
    sudo usermod -aG gpio $USER
    sudo chmod a+rw /sys/class/gpio/export
    # OR add Jetson GPIO udev rules (see Jetson.GPIO README)
    sudo busybox devmem 0x2430068 w 0x8
    sudo busybox devmem 0x2430070 w 0x8

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

---

## Configuration (`src/dms/config.py`)

Key values to tune for your driver:

| Constant             | Default | Meaning                               |
| -------------------- | ------- | ------------------------------------- |
| `EAR_THRESHOLD`      | 0.20    | Eye Aspect Ratio -> drowsy below this |
| `EAR_CONSEC_FRAMES`  | 3       | Frames EAR must stay low              |
| `MAR_THRESHOLD`      | 0.65    | Mouth Aspect Ratio -> yawn above this |
| `YAW_THRESHOLD`      | 30.0    | Head turn left/right limit            |
| `PITCH_THRESHOLD`    | 20.0    | Head up/down limit                    |
| `WARNING_DURATION_S` | 2.0     | Seconds in WARNING -> CRITICAL        |
| `ALERT_MOCK`         | True    | Set `False` for real GPIO             |

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

    # PFLD landmarks
    pdm run python tests/test_pfld.py

    # Head pose
    pdm run python tests/test_head_pose.py

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

                (any_event sustained >= WARNING_DURATION_S)
    NORMAL --------------------------------------------------> CRITICAL
      ^        (any_event appears)                                 |
      |   NORMAL ----------> WARNING ----------------------------> |
      |                        |  (event clears)                   |
      +------------------------+                                   |
      +----------------------------- (event clears) ---------------+

| State    | Green LED | Yellow LED | Red LED |  Vibration  |   Sound    |
| -------- | :-------: | :--------: | :-----: | :---------: | :--------: |
| NORMAL   |    on     |    off     |   off   |     off     |   silent   |
| WARNING  |    off    |     on     |   off   | short pulse |   silent   |
| CRITICAL |    off    |    off     |   on    | continuous  | alarm loop |

---

## Wiring Diagram

    Jetson 40-pin Header
    --------------------
    Pin 7  -- Green  LED anode  -> GND
    Pin 31 -- Yellow LED anode  -> GND
    Pin 29 -- Red    LED anode  -> GND
    Pin 33 -- MOSFET | -> Diode -> vibration motor -> 5 V
                     | -> GND
    USB Sound Card -- USB port -> Speaker / Buzzer
    USB Camera     -- USB port -> Raw Frames

> **Note** – Jetson BCM numbering may differ from Raspberry Pi.
> Verify actual physical pin locations with `pinmux` or the Jetson
> Orin Nano pinout diagram before wiring.

---

## How to Demo (Simulation Mode)

If you are grading this project on a machine without the custom Jetson GPIO pins, I2C sensors, or a USB camera connected, use this command to run a full demonstration.

It safely bypasses the hardware requirements and runs the full TensorRT AI pipeline on a pre-recorded sample video:

```bash
docker run -d \
  --name dms-core-sim \
  --runtime nvidia \
  --network host \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/models:/app/models \
  -v $(pwd)/test_video.mp4:/app/test_video.mp4 \
  -e HEALTHZ_PORT=8000 \
  ghcr.io/farhanhdaulay/capstone_project:latest


## Troubleshooting

| Symptom | Fix |
|---|---|
| GPIO permission denied | `sudo usermod -aG gpio $USER` then re-login |
| No USB audio | `aplay -l` -> set `defaults.pcm.card` in `/etc/asound.conf` |
| TensorRT engine fails | Rebuild with `model.export(format="engine")` on Jetson |
| Low FPS (<15) | Reduce `CAMERA_WIDTH/HEIGHT`, enable TRT engine, use `--no-window` |
| Face not detected | Improve lighting; lower `FACE_CONF_THRESHOLD` in config |
```

# tests/test_phone.py
# Step 5.5 — YOLOv11n Phone Detection Test
# Run with camera: pdm run python tests/test_phone.py
# Run with image:  pdm run python tests/test_phone.py --image data/sample_faces/phone.jpg
# Run mock:        pdm run python tests/test_phone.py --mock

import sys, time, argparse
import numpy as np
import cv2

sys.path.insert(0, 'src')
from dms.modules.phone import PhoneDetector

parser = argparse.ArgumentParser()
parser.add_argument('--model',  default='models/yolo26n.pt',
                    help='Path to YOLO weights (.pt or .engine)')
parser.add_argument('--image',  default=None)
parser.add_argument('--mock',   action='store_true')
parser.add_argument('--frames', type=int, default=20)
parser.add_argument('--source', default='csi')
args = parser.parse_args()

print(f"\n{'='*50}")
print(f"  Step 5.5 — YOLOv11n Phone Detection Test")
print(f"{'='*50}\n")

# ── 1. Load model ────────────────────────────────────────────────────
print(f"[1/5] Loading YOLO model: {args.model}")
try:
    detector = PhoneDetector(model_path=args.model)
    print("      ✅  Model loaded OK")
except Exception as e:
    print(f"      ❌  {e}")
    sys.exit(1)

# ── 2. COCO class check — 'cell phone' is class 67 ──────────────────
print("[2/5] Checking YOLO class names...")
try:
    names = detector.model.names
    phone_class_id = next((k for k,v in names.items()
                           if 'phone' in v.lower() or 'cell' in v.lower()), None)
    assert phone_class_id is not None, \
        "No 'cell phone' class found in model — wrong weights?"
    print(f"      ✅  'cell phone' → class ID {phone_class_id} ({names[phone_class_id]})")
except AttributeError:
    print("      ⚠️   Could not verify class names (may be TensorRT engine)")

# ── 3. Get a test frame ──────────────────────────────────────────────
print("[3/5] Acquiring test frame...")
if args.mock:
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 80
    # Draw a rough rectangle to simulate a phone-like object
    cv2.rectangle(frame, (250, 150), (390, 330), (200, 200, 200), -1)
    print("      [MOCK] Synthetic frame with rectangle (phone sim)")
elif args.image:
    frame = cv2.imread(args.image)
    assert frame is not None, f"Could not load: {args.image}"
    print(f"      [IMAGE] {args.image}: {frame.shape}")
else:
    from dms.modules.camera import CameraCapture
    cam = CameraCapture(source=args.source)
    assert cam.is_opened(), "Camera failed — try --mock or --image"
    ret, frame = cam.read()
    cam.release()
    assert ret, "Failed to read frame"
    print(f"      [CAMERA] Frame: {frame.shape}")

# ── 4. Inference + output validation ────────────────────────────────
print("[4/5] Inference test...")
detector.detect(frame)
t0 = time.time()
result = detector.detect(frame)
latency = (time.time() - t0) * 1000

print(f"      Result: {result}")
print(f"      Latency: {latency:.1f} ms")

required = ['phone_detected', 'phone_near_face', 'confidence', 'bbox', 'phone_sec']
for k in required:
    assert k in result, f"Missing key '{k}' in phone detector output"

assert isinstance(result['phone_detected'], bool), "'phone_detected' must be bool"
assert 0.0 <= result['confidence'] <= 1.0,         "'confidence' out of [0,1]"
assert latency < 2000, f"Phone detector too slow: {latency:.0f} ms"
print(f"      ✅  phone_detected={result['phone_detected']}  "
      f"confidence={result['confidence']:.2f}  latency={latency:.1f} ms")

# ── 5. Throughput ────────────────────────────────────────────────────
print(f"[5/5] Throughput test ({args.frames} frames)...")
times = []
for _ in range(args.frames):
    t0 = time.time()
    detector.detect(frame)
    times.append((time.time() - t0) * 1000)

avg_ms = np.mean(times)
fps = 1000 / avg_ms
print(f"      avg={avg_ms:.1f} ms/frame  →  {fps:.1f} FPS")
assert fps >= 2.0, f"Phone detector throughput too low: {fps:.1f} FPS"
print(f"      ✅  Throughput OK ({fps:.1f} FPS)")

print(f"\n{'='*50}")
print("  ✅  STEP 5.5 PHONE DETECTION TEST PASSED")
print(f"{'='*50}\n")

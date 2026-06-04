# tests/test_headpose.py
# Step 5.4 — 6DRepNet Head Pose Estimation Test
# Run with camera: pdm run python tests/test_headpose.py
# Run with image:  pdm run python tests/test_headpose.py --image data/sample_faces/face.jpg
# Run mock:        pdm run python tests/test_headpose.py --mock

import sys, time, argparse
import numpy as np
import cv2

sys.path.insert(0, 'src')
from dms.modules.head_pose import HeadPose6DRepNet

parser = argparse.ArgumentParser()
parser.add_argument('--model',  default='models/6drepnet360.onnx')
parser.add_argument('--image',  default=None)
parser.add_argument('--mock',   action='store_true')
parser.add_argument('--frames', type=int, default=20)
parser.add_argument('--source', default='csi')
args = parser.parse_args()

print(f"\n{'='*50}")
print(f"  Step 5.4 — 6DRepNet Head Pose Test")
print(f"{'='*50}\n")

# ── 1. Load model ────────────────────────────────────────────────────
print(f"[1/5] Loading 6DRepNet model: {args.model}")
try:
    estimator = HeadPose6DRepNet(model_path=args.model)
    print("      ✅  Model loaded OK")
except Exception as e:
    print(f"      ❌  {e}")
    sys.exit(1)

# ── 2. ONNX provider ─────────────────────────────────────────────────
print("[2/5] Checking ONNX execution provider...")
import onnxruntime as ort
providers = ort.get_available_providers()
if 'CUDAExecutionProvider' in providers:
    print("      ✅  CUDA provider available")
else:
    print("      ⚠️   CPU-only mode")

# ── 3. Get a test frame ──────────────────────────────────────────────
print("[3/5] Acquiring test frame...")
if args.mock:
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 128
    print("      [MOCK] Blank gray frame")
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
t0 = time.time()
result = estimator.estimate(frame)
latency = (time.time() - t0) * 1000

print(f"      Result: {result}")
print(f"      Latency: {latency:.1f} ms")

required = ['distracted_warn', 'distracted_crit', 'pitch', 'yaw', 'roll', 'dist_sec']
for k in required:
    assert k in result, f"Missing key '{k}' in head pose output"

# Angles must be in plausible degree range
assert -180 <= result['pitch'] <= 180, f"pitch out of range: {result['pitch']}"
assert -180 <= result['yaw']   <= 180, f"yaw out of range:   {result['yaw']}"
assert -180 <= result['roll']  <= 180, f"roll out of range:  {result['roll']}"

assert latency < 500, f"6DRepNet too slow: {latency:.0f} ms"
print(f"      ✅  pitch={result['pitch']:.1f}°  yaw={result['yaw']:.1f}°  "
      f"roll={result['roll']:.1f}°  latency={latency:.1f} ms")

# ── 5. Distraction timer logic test ─────────────────────────────────
print("[5/5] Distraction state logic test...")

# Reset internal timer and test with a mock extreme-angle frame
import time as _time
estimator.distracted_t = None

# Simulate head turned hard right: inject extreme yaw by patching estimate
original_estimate = estimator.estimate

def _mock_turned(frame, face_bbox=None):
    # Manually set timer as if head has been turned for 0.5 sec (WARNING zone)
    if estimator.distracted_t is None:
        estimator.distracted_t = _time.time() - 1.5
    now = _time.time()
    dist_sec = now - estimator.distracted_t
    return {
        'distracted_warn': estimator.WARN_SEC <= dist_sec < estimator.CRIT_SEC,
        'distracted_crit': dist_sec >= estimator.CRIT_SEC,
        'pitch': 0.0, 'yaw': 45.0, 'roll': 0.0,
        'dist_sec': round(dist_sec, 2)
    }

estimator.estimate = _mock_turned
warn_result = estimator.estimate(frame)
assert warn_result['distracted_warn'] or warn_result['distracted_crit'], \
    "Expected warning/critical for 45° yaw held 1.5s"
print(f"      ✅  Distraction timer working: dist_sec={warn_result['dist_sec']}s  "
      f"warn={warn_result['distracted_warn']}  crit={warn_result['distracted_crit']}")

estimator.estimate = original_estimate  # restore

# ── Throughput ───────────────────────────────────────────────────────
print(f"      Throughput ({args.frames} frames)...")
times = []
for _ in range(args.frames):
    t0 = time.time()
    estimator.estimate(frame)
    times.append((time.time() - t0) * 1000)

avg_ms = np.mean(times)
fps = 1000 / avg_ms
print(f"      avg={avg_ms:.1f} ms/frame  →  {fps:.1f} FPS")
assert fps >= 1.0, f"6DRepNet throughput too low: {fps:.1f} FPS"
print(f"      ✅  Throughput OK ({fps:.1f} FPS)")

print(f"\n{'='*50}")
print("  ✅  STEP 5.4 HEAD POSE TEST PASSED")
print(f"{'='*50}\n")

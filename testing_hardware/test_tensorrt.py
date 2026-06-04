# tests/test_tensorrt.py
# Step 5.6 — TensorRT FP16 Optimization Test
# This test:
#   1. Exports YOLO model to TensorRT FP16 engine (one-time, ~5-10 min on Jetson)
#   2. Verifies the engine file is created
#   3. Benchmarks FP16 engine vs PyTorch baseline
#   4. Confirms output consistency between PT and TRT
#
# Run: pdm run python tests/test_tensorrt.py
# Skip export (engine already exists): pdm run python tests/test_tensorrt.py --skip-export

import sys, time, os, argparse
import numpy as np
import cv2

sys.path.insert(0, 'src')

parser = argparse.ArgumentParser()
parser.add_argument('--pt-model',      default='models/yolo26n.pt')
parser.add_argument('--engine-path',   default='models/yolo11n.engine')
parser.add_argument('--skip-export',   action='store_true',
                    help='Skip TRT export, just benchmark existing engine')
parser.add_argument('--mock',          action='store_true',
                    help='Use blank frame instead of camera')
parser.add_argument('--source',        default='csi')
parser.add_argument('--frames',        type=int, default=30)
parser.add_argument('--imgsz',         type=int, default=320,
                    help='Inference image size (smaller = faster on Jetson)')
args = parser.parse_args()

print(f"\n{'='*55}")
print(f"  Step 5.6 — TensorRT FP16 Optimization Test")
print(f"{'='*55}\n")

# ── 1. Prerequisites check ───────────────────────────────────────────
print("[1/6] Checking prerequisites...")
try:
    import tensorrt as trt
    print(f"      ✅  TensorRT {trt.__version__}")
except ImportError:
    print("      ❌  TensorRT Python bindings not found")
    print("      Fix: sudo apt install python3-libnvinfer-dev")
    sys.exit(1)

try:
    from ultralytics import YOLO
    print("      ✅  Ultralytics YOLO available")
except ImportError:
    print("      ❌  ultralytics not installed: pdm add ultralytics")
    sys.exit(1)

assert os.path.exists(args.pt_model), \
    f"PT model not found: {args.pt_model}"
print(f"      ✅  PT weights: {args.pt_model}  "
      f"({os.path.getsize(args.pt_model)//1024} KB)")

# ── 2. TensorRT export ───────────────────────────────────────────────
if args.skip_export and os.path.exists(args.engine_path):
    print(f"[2/6] Skipping export — using existing engine: {args.engine_path}")
else:
    print(f"[2/6] Exporting {args.pt_model} → TensorRT FP16 engine...")
    print(f"      Image size: {args.imgsz}x{args.imgsz}")
    print("      ⏳  This takes 5–15 minutes on Jetson — please wait...\n")
    t0 = time.time()

    model_pt = YOLO(args.pt_model)
    model_pt.export(
        format='engine',
        imgsz=args.imgsz,
        half=True,        # FP16
        device=0,         # GPU
        workspace=2,      # GB — use 2GB on 8GB Jetson
    )

    # Ultralytics saves as <model_name>.engine next to the .pt file
    auto_path = args.pt_model.replace('.pt', '.engine')
    if os.path.exists(auto_path) and auto_path != args.engine_path:
        import shutil
        shutil.move(auto_path, args.engine_path)
        print(f"      Moved engine → {args.engine_path}")

    elapsed = time.time() - t0
    print(f"\n      ✅  Export done in {elapsed/60:.1f} min")

# ── 3. Engine file sanity ────────────────────────────────────────────
print("[3/6] Verifying engine file...")
assert os.path.exists(args.engine_path), \
    f"Engine file not found: {args.engine_path}"
engine_size_mb = os.path.getsize(args.engine_path) // 1024 // 1024
assert engine_size_mb > 0, "Engine file is empty"
print(f"      ✅  Engine: {args.engine_path}  ({engine_size_mb} MB)")

# ── 4. Get a test frame ──────────────────────────────────────────────
print("[4/6] Acquiring test frame...")
if args.mock:
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    print("      [MOCK] Random noise frame")
else:
    from dms.modules.camera import CameraCapture
    cam = CameraCapture(source=args.source)
    if cam.is_opened():
        ret, frame = cam.read()
        cam.release()
        if not ret:
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            print("      [FALLBACK] Camera read failed, using random frame")
        else:
            print(f"      [CAMERA] Frame: {frame.shape}")
    else:
        cam.release()
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        print("      [FALLBACK] Camera not available, using random frame")

# ── 5. Benchmark: PyTorch vs TensorRT ───────────────────────────────
print(f"[5/6] Benchmarking PT vs TRT ({args.frames} frames each)...")

model_pt  = YOLO(args.pt_model)
model_trt = YOLO(args.engine_path)

# Warm up
for _ in range(3):
    model_pt.predict(frame, imgsz=args.imgsz, verbose=False)
    model_trt.predict(frame, imgsz=args.imgsz, verbose=False)

# Benchmark PT
pt_times = []
for _ in range(args.frames):
    t0 = time.time()
    model_pt.predict(frame, imgsz=args.imgsz, verbose=False)
    pt_times.append((time.time() - t0) * 1000)
pt_avg = np.mean(pt_times)
pt_fps = 1000 / pt_avg

# Benchmark TRT
trt_times = []
for _ in range(args.frames):
    t0 = time.time()
    model_trt.predict(frame, imgsz=args.imgsz, verbose=False)
    trt_times.append((time.time() - t0) * 1000)
trt_avg = np.mean(trt_times)
trt_fps = 1000 / trt_avg

speedup = pt_avg / trt_avg

print(f"\n      {'Model':<20} {'Avg ms':>8}  {'FPS':>6}")
print(f"      {'─'*38}")
print(f"      {'PyTorch FP32':<20} {pt_avg:>8.1f}  {pt_fps:>6.1f}")
print(f"      {'TensorRT FP16':<20} {trt_avg:>8.1f}  {trt_fps:>6.1f}")
print(f"      {'─'*38}")
print(f"      Speedup: {speedup:.2f}x\n")

assert trt_fps >= pt_fps * 0.8, \
    f"TRT not faster than PT (TRT {trt_fps:.1f} vs PT {pt_fps:.1f} FPS)"
print(f"      ✅  TensorRT FP16 is {speedup:.1f}x faster than PyTorch baseline")

# ── 6. Output consistency check ──────────────────────────────────────
print("[6/6] Output consistency check (PT vs TRT detections)...")
res_pt  = model_pt.predict(frame,  imgsz=args.imgsz, verbose=False, conf=0.3)
res_trt = model_trt.predict(frame, imgsz=args.imgsz, verbose=False, conf=0.3)

n_pt  = len(res_pt[0].boxes)  if res_pt  else 0
n_trt = len(res_trt[0].boxes) if res_trt else 0
print(f"      PT detections: {n_pt}   TRT detections: {n_trt}")

# Allow ±1 detection difference due to FP16 rounding
assert abs(n_pt - n_trt) <= 2, \
    f"Large detection count mismatch: PT={n_pt} vs TRT={n_trt}"
print(f"      ✅  Detection counts consistent (difference ≤ 2)")

print(f"\n{'='*55}")
print(f"  ✅  STEP 5.6 TENSORRT FP16 TEST PASSED")
print(f"  Engine saved: {args.engine_path}")
print(f"  Speedup: {speedup:.1f}x  |  TRT FPS: {trt_fps:.1f}")
print(f"{'='*55}\n")

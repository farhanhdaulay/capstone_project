# tests/test_camera.py
# Step 5.1 — Video Capture Module Test
# Run: pdm run python tests/test_camera.py
# Run with USB fallback: pdm run python tests/test_camera.py --source 0

import sys, time, argparse
import cv2

# ── Allow running from project root without installing the package ──
sys.path.insert(0, 'src')
from dms.modules.camera import CameraCapture

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='csi',
                    help="'csi' for IMX219, or integer index for USB (e.g. 0)")
parser.add_argument('--frames', type=int, default=30)
args = parser.parse_args()

print(f"\n{'='*50}")
print(f"  Step 5.1 — Camera Test  (source={args.source})")
print(f"{'='*50}\n")

# ── 1. Open camera ──────────────────────────────────────────────────
print("[1/4] Opening camera...")
cam = CameraCapture(source=args.source)
assert cam.is_opened(), (
    "Camera failed to open.\n"
    "  CSI: check /dev/video0 exists and nvargus-daemon is running.\n"
    "  USB: try --source 0"
)
print("      ✅  Camera opened OK")

# ── 2. Read frames and check shape ──────────────────────────────────
print(f"[2/4] Reading {args.frames} frames...")
success, failures = 0, 0
t0 = time.time()

for i in range(args.frames):
    ret, frame = cam.read()
    if ret and frame is not None:
        h, w, c = frame.shape
        assert c == 3,  f"Expected 3 channels (BGR), got {c}"
        assert h > 0 and w > 0, f"Invalid frame size {w}x{h}"
        success += 1
        if i == 0:
            print(f"      First frame: {w}x{h}, dtype={frame.dtype}")
    else:
        failures += 1

elapsed = time.time() - t0
fps = success / elapsed if elapsed > 0 else 0

print(f"      ✅  {success}/{args.frames} frames OK  ({fps:.1f} FPS,  {failures} failures)")

# ── 3. FPS sanity check ─────────────────────────────────────────────
print("[3/4] FPS check...")
assert success >= args.frames * 0.9, \
    f"Too many frame failures: {failures}/{args.frames}"
assert fps >= 5.0, \
    f"FPS too low: {fps:.1f} (expected ≥ 5 for DMS pipeline)"
print(f"      ✅  FPS {fps:.1f} is within acceptable range (≥5 FPS)")

# ── 4. Release ──────────────────────────────────────────────────────
print("[4/4] Releasing camera...")
cam.release()
print("      ✅  Released OK")

print(f"\n{'='*50}")
print("  ✅  STEP 5.1 CAMERA TEST PASSED")
print(f"{'='*50}\n")

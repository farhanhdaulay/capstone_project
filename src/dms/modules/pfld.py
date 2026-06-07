# src/dms/modules/pfld.py
"""
PFLD landmark detector - EAR and MAR computation.
Expects a face bounding box from FaceDetector before running.

Supports:
  - 68-point model  (136 outputs)  <-- pfld_106_lite.onnx actually outputs 136
  - 106-point model (212 outputs)
"""

import cv2
import numpy as np
from dms.modules.trt_backend import TRTWrapper
from dms.config import PFLD_MODEL


# ---------------------------------------------------------------------------
# Index sets per model variant
# ---------------------------------------------------------------------------

# 68-point iBUG 300-W standard indices (0-based)
# Eye layout: [outer_corner, upper1, upper2, inner_corner, lower2, lower1]
# EAR formula uses vertical pairs (p1,p5)(p2,p4) over horizontal span (p0,p3)
_LEFT_EYE_68  = [36, 37, 38, 39, 40, 41]
_RIGHT_EYE_68 = [42, 43, 44, 45, 46, 47]
_MOUTH_TOP_68 = 51
_MOUTH_BOT_68 = 57
_MOUTH_L_68   = 48
_MOUTH_R_68   = 54

# 106-point PFLD-lite indices
_LEFT_EYE_106  = [60, 61, 62, 63, 64, 65]
_RIGHT_EYE_106 = [68, 69, 70, 71, 72, 73]
_MOUTH_TOP_106 = 84
_MOUTH_BOT_106 = 90
_MOUTH_L_106   = 76
_MOUTH_R_106   = 82


def _euc(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def draw_landmarks(frame, landmarks, colour=(0, 220, 255), radius=2):
    """Draw landmark dots on frame in-place. Bright cyan, radius=2."""
    for (lx, ly) in landmarks:
        cv2.circle(frame, (int(lx), int(ly)), radius, colour, -1)


class PFLDDetector:
    """
    Runs PFLD on a pre-cropped face ROI using TensorRT.

    Usage:
        detector = PFLDDetector()
        result   = detector.run(frame, face_bbox)
        # result["ear"], result["mar"], result["landmarks"]
    """

    INPUT_SIZE = (112, 112)

    def __init__(self, model_path=PFLD_MODEL):
        # 1. Initialize the custom TensorRT wrapper
        self.sess = TRTWrapper(model_path)

        # 2. Extract shape from the wrapper's pre-allocated outputs
        out_shape = self.sess.outputs[0]['shape']
        raw_dim   = out_shape[-1]

        # 3. Handle dynamic vs static shapes
        if isinstance(raw_dim, int) and raw_dim > 1:
            self.n_pts = raw_dim // 2
        else:
            # Dynamic batch dim: run a dummy pass to get the real size
            dummy = np.zeros((1, 3, 112, 112), dtype=np.float32)
            dummy = np.ascontiguousarray(dummy) # MUST be contiguous for TRT
            out   = self.sess.predict(dummy)[0]
            self.n_pts = out.shape[-1] // 2

        print("[PFLD] Loaded TRT Engine: {}".format(model_path))
        print("       output shape = {}".format(out_shape))
        print("       landmarks = {}".format(self.n_pts))

        # Pick correct index set based on what the model actually outputs
        if self.n_pts == 106:
            self._left_eye  = _LEFT_EYE_106
            self._right_eye = _RIGHT_EYE_106
            self._mouth_top = _MOUTH_TOP_106
            self._mouth_bot = _MOUTH_BOT_106
            self._mouth_l   = _MOUTH_L_106
            self._mouth_r   = _MOUTH_R_106
            print("       index set = 106-point PFLD-lite")
        else:
            # 68-point iBUG 300-W (model outputs 136 values)
            self._left_eye  = _LEFT_EYE_68
            self._right_eye = _RIGHT_EYE_68
            self._mouth_top = _MOUTH_TOP_68
            self._mouth_bot = _MOUTH_BOT_68
            self._mouth_l   = _MOUTH_L_68
            self._mouth_r   = _MOUTH_R_68
            print("       index set = 68-point iBUG 300-W")
            print("       L-eye={} R-eye={}".format(self._left_eye, self._right_eye))

    # -- public API ------------------------------------------------------------

    def run(self, frame, face_bbox):
        """
        Args:
            frame:     full BGR frame (H x W x 3)
            face_bbox: (x1, y1, x2, y2) pixel coords

        Returns dict with keys:
            ear        float
            mar        float
            landmarks  list of (x, y) in full-frame pixel coords
        """
        x1, y1, x2, y2 = [int(v) for v in face_bbox]

        # Clamp to frame bounds
        fh, fw = frame.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(fw, x2)
        y2 = min(fh, y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return {"ear": None, "mar": None, "landmarks": []}

        roi_h, roi_w = roi.shape[:2]

        # Preprocess
        img = cv2.resize(roi, self.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = img.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, 112, 112)

        # Ensure contiguous memory before PyCUDA transfer!
        inp = np.ascontiguousarray(inp)

        # Inference: TRTWrapper returns a list of outputs. Grab the first one.
        raw = self.sess.predict(inp)[0]
        raw = raw.reshape(-1)  # flatten: (136,) or (212,)

        # Decode: normalised [0,1] -> ROI pixels -> full-frame pixels
        pts = [
            (float(raw[i * 2])     * roi_w + x1,
             float(raw[i * 2 + 1]) * roi_h + y1)
            for i in range(self.n_pts)
        ]

        ear = (self._ear(pts, self._left_eye) +
               self._ear(pts, self._right_eye)) / 2.0
        mar = self._mar(pts)

        return {
            "ear":       round(ear, 3),
            "mar":       round(mar, 3),
            "landmarks": pts,
        }

    # -- internals -------------------------------------------------------------

    def _ear(self, pts, idx):
        """Eye Aspect Ratio - Soukupova and Cech (2016)."""
        p = [pts[i] for i in idx]
        return ((_euc(p[1], p[5]) + _euc(p[2], p[4])) /
                (2.0 * _euc(p[0], p[3]) + 1e-6))

    def _mar(self, pts):
        """Mouth Aspect Ratio."""
        return (_euc(pts[self._mouth_top], pts[self._mouth_bot]) /
                (_euc(pts[self._mouth_l],  pts[self._mouth_r])   + 1e-6))
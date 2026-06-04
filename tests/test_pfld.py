"""
tests/test_pfld.py
==================
Coverage target: = 90% of src/dms/modules/pfld.py

onnxruntime.InferenceSession is fully mocked -- no .onnx file required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_ORT_SESSION = "dms.modules.pfld.ort.InferenceSession"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fake_landmarks(n_pts: int) -> np.ndarray:
    """Return a flat array of n_pts * 2 normalised coords (all 0.5)."""
    return np.full((1, n_pts * 2), 0.5, dtype=np.float32)


def _make_session(static_dim, runtime_out=None):
    sess = MagicMock()
    sess.get_providers.return_value = ["CPUExecutionProvider"]

    inp       = MagicMock()
    inp.name  = "input"
    sess.get_inputs.return_value = [inp]

    out        = MagicMock()
    out.name   = "output"
    out.shape  = ["batch", static_dim] if isinstance(static_dim, int) else ["batch_size", "dyn"]
    sess.get_outputs.return_value = [out]

    if runtime_out is None:
        n = static_dim if isinstance(static_dim, int) else 212
        runtime_out = _fake_landmarks(n // 2)
    sess.run.return_value = [runtime_out]
    return sess


def _make_pfld(n_pts=106):
    static_dim  = n_pts * 2
    runtime_out = _fake_landmarks(n_pts)
    sess = _make_session(static_dim, runtime_out)
    with patch(_ORT_SESSION, return_value=sess), \
         patch("dms.modules.pfld.PFLD_MODEL", "fake.onnx"):
        from dms.modules.pfld import PFLDDetector
        det = PFLDDetector(model_path="fake.onnx")
    det.sess = sess
    return det


def _make_pfld_dynamic():
    """Dynamic output dim -> runtime probe path."""
    runtime_out = _fake_landmarks(106)
    sess = _make_session(static_dim="dyn", runtime_out=runtime_out)
    with patch(_ORT_SESSION, return_value=sess), \
         patch("dms.modules.pfld.PFLD_MODEL", "fake.onnx"):
        from dms.modules.pfld import PFLDDetector
        det = PFLDDetector(model_path="fake.onnx")
    det.sess = sess
    return det


# ---------------------------------------------------------------------------
# 1. __init__ -- landmark index selection
# ---------------------------------------------------------------------------

class TestInit:
    def test_106_point_index_set(self):
        det = _make_pfld(106)
        assert det.n_pts == 106
        from dms.modules.pfld import _LEFT_EYE_106
        assert det._left_eye == _LEFT_EYE_106

    def test_68_point_index_set(self):
        det = _make_pfld(68)
        assert det.n_pts == 68
        from dms.modules.pfld import _LEFT_EYE_68
        assert det._left_eye == _LEFT_EYE_68

    def test_dynamic_dim_resolved_via_runtime(self):
        det = _make_pfld_dynamic()
        assert det.n_pts == 106


# ---------------------------------------------------------------------------
# 2. run() -- empty ROI guard
# ---------------------------------------------------------------------------

class TestRunEmpty:
    def test_zero_size_roi_returns_none_ear_mar(self):
        det   = _make_pfld(106)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = det.run(frame, (50, 50, 50, 50))   # empty
        assert result["ear"]       is None
        assert result["mar"]       is None
        assert result["landmarks"] == []

    def test_out_of_bounds_bbox_clamped(self):
        det   = _make_pfld(106)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # huge bbox -> clamped to frame, should not crash
        result = det.run(frame, (-50, -50, 200, 200))
        assert "ear" in result


# ---------------------------------------------------------------------------
# 3. run() -- 106-point path
# ---------------------------------------------------------------------------

class TestRun106:
    def test_returns_ear_mar_landmarks(self):
        det   = _make_pfld(106)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = det.run(frame, (10, 10, 190, 190))
        assert "ear" in result
        assert "mar" in result
        assert len(result["landmarks"]) == 106

    def test_ear_is_positive(self):
        det   = _make_pfld(106)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = det.run(frame, (10, 10, 190, 190))
        assert result["ear"] >= 0.0

    def test_landmarks_in_frame_coords(self):
        """Landmarks should be scaled to full-frame pixel space, not [0,1]."""
        det   = _make_pfld(106)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = det.run(frame, (10, 10, 190, 190))
        # All coordinates should be >= 10 (offset by x1/y1)
        for x, y in result["landmarks"]:
            assert x >= 10
            assert y >= 10


# ---------------------------------------------------------------------------
# 4. run() -- 68-point path
# ---------------------------------------------------------------------------

class TestRun68:
    def test_returns_ear_mar_landmarks(self):
        det   = _make_pfld(68)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = det.run(frame, (10, 10, 190, 190))
        assert len(result["landmarks"]) == 68
        assert result["ear"] >= 0.0
        assert result["mar"] >= 0.0


# ---------------------------------------------------------------------------
# 5. _ear() and _mar() directly
# ---------------------------------------------------------------------------

class TestEarMar:
    def _uniform_pts(self, n=106):
        """Points all at (100, 100) -- EAR should be ~0 (numerator zero)."""
        return [(100.0, 100.0)] * n

    def test_ear_zero_when_all_same_point(self):
        det = _make_pfld(106)
        pts = self._uniform_pts(106)
        from dms.modules.pfld import _LEFT_EYE_106
        ear = det._ear(pts, _LEFT_EYE_106)
        assert ear == pytest.approx(0.0, abs=1e-5)

    def test_mar_zero_when_all_same_point(self):
        det = _make_pfld(106)
        pts = self._uniform_pts(106)
        mar = det._mar(pts)
        assert mar == pytest.approx(0.0, abs=1e-5)

    def test_ear_positive_for_open_eye(self):
        """Construct an eye where vertical distance > 0."""
        det = _make_pfld(106)
        from dms.modules.pfld import _LEFT_EYE_106
        pts = [(0.0, 0.0)] * 106
        # p0=outer, p3=inner (horizontal), p1,p5 and p2,p4 (vertical)
        idx = _LEFT_EYE_106
        pts[idx[0]] = (0.0,  0.0)    # outer corner
        pts[idx[3]] = (10.0, 0.0)    # inner corner  (horizontal span = 10)
        pts[idx[1]] = (3.0,  3.0)    # upper1
        pts[idx[2]] = (7.0,  3.0)    # upper2
        pts[idx[4]] = (7.0, -3.0)    # lower2
        pts[idx[5]] = (3.0, -3.0)    # lower1
        ear = det._ear(pts, idx)
        assert ear > 0.0


# ---------------------------------------------------------------------------
# 6. draw_landmarks()
# ---------------------------------------------------------------------------

class TestDrawLandmarks:
    def test_draw_does_not_raise(self):
        from dms.modules.pfld import draw_landmarks
        import cv2
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        lms   = [(50.0, 60.0), (70.0, 80.0)]
        with patch.object(cv2, "circle"):
            draw_landmarks(frame, lms)   # should not raise

    def test_draw_calls_circle_per_point(self):
        from dms.modules.pfld import draw_landmarks
        import cv2
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        lms   = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
        with patch.object(cv2, "circle") as mock_circle:
            draw_landmarks(frame, lms)
            assert mock_circle.call_count == 3
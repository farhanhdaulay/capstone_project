"""
tests/test_pfld.py
==================
Coverage target: = 90% of src/dms/modules/pfld.py
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

# FAKE THE HARDWARE LIBRARIES BEFORE IMPORTING DMS MODULES
sys.modules["tensorrt"] = MagicMock()
sys.modules["pycuda"] = MagicMock()
sys.modules["pycuda.driver"] = MagicMock()
sys.modules["pycuda.autoinit"] = MagicMock()

def _fake_landmarks(n_pts: int) -> np.ndarray:
    return np.full((1, n_pts * 2), 0.5, dtype=np.float32)

def _make_pfld(n_pts=106):
    static_dim = n_pts * 2
    runtime_out = _fake_landmarks(n_pts)
    
    mock_wrapper = MagicMock()
    mock_wrapper.outputs = [{'shape': ["batch", static_dim]}]
    mock_wrapper.predict.return_value = [runtime_out]

    with patch("dms.modules.pfld.TRTWrapper", return_value=mock_wrapper), \
         patch("dms.modules.pfld.PFLD_MODEL", "fake.engine"):
        from dms.modules.pfld import PFLDDetector
        det = PFLDDetector(model_path="fake.engine")
    return det

def _make_pfld_dynamic():
    runtime_out = _fake_landmarks(106)
    mock_wrapper = MagicMock()
    mock_wrapper.outputs = [{'shape': ["batch", "dynamic"]}]
    mock_wrapper.predict.return_value = [runtime_out]

    with patch("dms.modules.pfld.TRTWrapper", return_value=mock_wrapper), \
         patch("dms.modules.pfld.PFLD_MODEL", "fake.engine"):
        from dms.modules.pfld import PFLDDetector
        det = PFLDDetector(model_path="fake.engine")
    return det

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

class TestRunEmpty:
    def test_zero_size_roi_returns_none_ear_mar(self):
        det   = _make_pfld(106)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = det.run(frame, (50, 50, 50, 50))
        assert result["ear"]       is None
        assert result["mar"]       is None
        assert result["landmarks"] == []

    def test_out_of_bounds_bbox_clamped(self):
        det   = _make_pfld(106)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = det.run(frame, (-50, -50, 200, 200))
        assert "ear" in result

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
        det   = _make_pfld(106)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = det.run(frame, (10, 10, 190, 190))
        for x, y in result["landmarks"]:
            assert x >= 10
            assert y >= 10

class TestRun68:
    def test_returns_ear_mar_landmarks(self):
        det   = _make_pfld(68)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = det.run(frame, (10, 10, 190, 190))
        assert len(result["landmarks"]) == 68
        assert result["ear"] >= 0.0
        assert result["mar"] >= 0.0

class TestEarMar:
    def _uniform_pts(self, n=106):
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
        det = _make_pfld(106)
        from dms.modules.pfld import _LEFT_EYE_106
        pts = [(0.0, 0.0)] * 106
        idx = _LEFT_EYE_106
        pts[idx[0]] = (0.0,  0.0)    
        pts[idx[3]] = (10.0, 0.0)    
        pts[idx[1]] = (3.0,  3.0)    
        pts[idx[2]] = (7.0,  3.0)    
        pts[idx[4]] = (7.0, -3.0)    
        pts[idx[5]] = (3.0, -3.0)    
        ear = det._ear(pts, idx)
        assert ear > 0.0

class TestDrawLandmarks:
    def test_draw_does_not_raise(self):
        from dms.modules.pfld import draw_landmarks
        import cv2
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        lms   = [(50.0, 60.0), (70.0, 80.0)]
        with patch.object(cv2, "circle"):
            draw_landmarks(frame, lms)

    def test_draw_calls_circle_per_point(self):
        from dms.modules.pfld import draw_landmarks
        import cv2
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        lms   = [(10.0, 20.0), (30.0, 40.0), (50.0, 60.0)]
        with patch.object(cv2, "circle") as mock_circle:
            draw_landmarks(frame, lms)
            assert mock_circle.call_count == 3
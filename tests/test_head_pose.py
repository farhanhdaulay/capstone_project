"""
tests/test_head_pose.py
=======================
Coverage target: = 90% of src/dms/modules/head_pose.py
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch
import numpy as np

# FAKE THE HARDWARE LIBRARIES BEFORE IMPORTING DMS MODULES
sys.modules["tensorrt"] = MagicMock()
sys.modules["pycuda"] = MagicMock()
sys.modules["pycuda.driver"] = MagicMock()
sys.modules["pycuda.autoinit"] = MagicMock()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_estimator(out_shape, runtime_out=None):
    mock_wrapper = MagicMock()
    mock_wrapper.outputs = [{'shape': out_shape}]
    
    if runtime_out is None:
        runtime_out = np.zeros((1, 6), dtype=np.float32)
    mock_wrapper.predict.return_value = [runtime_out]

    with patch("dms.modules.head_pose.TRTWrapper", return_value=mock_wrapper), \
         patch("dms.modules.head_pose.HEAD_POSE_MODEL", "fake.engine"):
        from dms.modules.head_pose import HeadPoseEstimator
        est = HeadPoseEstimator(model_path="fake.engine")
    return est

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestInit:
    def test_detects_6d_from_static_shape(self):
        est = _make_estimator(out_shape=["batch", 6])
        assert est._out_format == "6d"

    def test_detects_rotmat_from_static_shape(self):
        rot = np.zeros((1, 3, 3), dtype=np.float32)
        est = _make_estimator(out_shape=["batch", 3, 3], runtime_out=rot)
        assert est._out_format == "rotmat"

class TestRunEmptyROI:
    def test_zero_size_roi_returns_zeros(self):
        est = _make_estimator(out_shape=["batch", 6])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = est.run(frame, (50, 50, 50, 50))
        assert result == {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    def test_out_of_bounds_bbox_clamped(self):
        est = _make_estimator(out_shape=["batch", 6])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = est.run(frame, (-10, -10, 200, 200))
        assert "pitch" in result

class TestRun6D:
    def test_returns_pitch_yaw_roll_keys(self):
        vec = np.array([[1, 0, 0, 0, 1, 0]], dtype=np.float32)
        est = _make_estimator(out_shape=["batch", 6], runtime_out=vec)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = est.run(frame, (10, 10, 190, 190))
        assert set(result.keys()) == {"pitch", "yaw", "roll"}

    def test_values_are_floats(self):
        vec = np.random.randn(1, 6).astype(np.float32)
        est = _make_estimator(out_shape=["batch", 6], runtime_out=vec)
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = est.run(frame, (10, 10, 190, 190))
        for k in ("pitch", "yaw", "roll"):
            assert isinstance(result[k], float)

class TestRunRotMat:
    def test_rotmat_path_returns_angles(self):
        R = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
        est = _make_estimator(out_shape=["batch", 3, 3], runtime_out=R)
        est._out_format = "rotmat"
        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = est.run(frame, (10, 10, 190, 190))
        assert set(result.keys()) == {"pitch", "yaw", "roll"}

class TestStaticMethods:
    def test_6d_to_rot_produces_orthonormal_matrix(self):
        from dms.modules.head_pose import HeadPoseEstimator
        v = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        R = HeadPoseEstimator._6d_to_rot(v)
        assert R.shape == (3, 3)
        for col in range(3):
            assert abs(np.linalg.norm(R[:, col]) - 1.0) < 1e-5

    def test_rot_to_euler_identity(self):
        from dms.modules.head_pose import HeadPoseEstimator
        R = np.eye(3)
        pitch, yaw, roll = HeadPoseEstimator._rot_to_euler(R)
        assert abs(pitch) < 1e-3
        assert abs(yaw)   < 1e-3
        assert abs(roll)  < 1e-3

    def test_rot_to_euler_singular_case(self):
        from dms.modules.head_pose import HeadPoseEstimator
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        pitch, yaw, roll = HeadPoseEstimator._rot_to_euler(R)
        assert isinstance(pitch, float)
        assert roll == 0.0

    def test_6d_to_rot_near_zero_norm_safe(self):
        from dms.modules.head_pose import HeadPoseEstimator
        v = np.zeros(6, dtype=np.float64)
        R = HeadPoseEstimator._6d_to_rot(v)
        assert R.shape == (3, 3)
        assert not np.any(np.isnan(R))
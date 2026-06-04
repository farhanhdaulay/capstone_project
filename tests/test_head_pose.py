"""
tests/test_head_pose.py
=======================
Coverage target: = 90% of src/dms/modules/head_pose.py

onnxruntime.InferenceSession is fully mocked -- no .onnx file required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

_ORT_SESSION = "dms.modules.head_pose.ort.InferenceSession"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_session(out_shape, runtime_out=None):
    """
    Build a mock ort.InferenceSession.

    out_shape    : list returned by get_outputs()[0].shape
    runtime_out  : value returned by session.run(); auto-built if None
    """
    sess = MagicMock()
    sess.get_providers.return_value = ["CPUExecutionProvider"]

    inp_meta       = MagicMock()
    inp_meta.name  = "input"
    inp_meta.shape = ["batch", 3, 224, 224]
    sess.get_inputs.return_value = [inp_meta]

    out_meta       = MagicMock()
    out_meta.name  = "output"
    out_meta.shape = out_shape
    sess.get_outputs.return_value = [out_meta]

    if runtime_out is None:
        runtime_out = np.zeros((1, 6), dtype=np.float32)
    sess.run.return_value = [runtime_out]
    return sess


def _make_estimator(out_shape, runtime_out=None):
    sess = _make_session(out_shape, runtime_out)
    with patch(_ORT_SESSION, return_value=sess), \
         patch("dms.modules.head_pose.HEAD_POSE_MODEL", "fake.onnx"):
        from dms.modules.head_pose import HeadPoseEstimator
        est = HeadPoseEstimator(model_path="fake.onnx")
    est.sess = sess
    return est


# ---------------------------------------------------------------------------
# 1. __init__ -- format detection
# ---------------------------------------------------------------------------

class TestInit:
    def test_detects_6d_from_static_shape(self):
        est = _make_estimator(out_shape=["batch", 6])
        assert est._out_format == "6d"

    def test_detects_rotmat_from_static_shape(self):
        rot = np.zeros((1, 3, 3), dtype=np.float32)
        est = _make_estimator(out_shape=["batch", 3, 3], runtime_out=rot)
        assert est._out_format == "rotmat"

    def test_detects_6d_via_runtime_probe(self):
        """Dynamic dim ('batch_size' string) -> runtime probe, shape ends in 6."""
        dummy_out = np.zeros((1, 6), dtype=np.float32)
        est = _make_estimator(out_shape=["batch_size", "dynamic"], runtime_out=dummy_out)
        assert est._out_format == "6d"

    def test_detects_rotmat_via_runtime_probe(self):
        dummy_out = np.zeros((1, 9), dtype=np.float32)   # not 6
        est = _make_estimator(out_shape=["batch_size", "dynamic"], runtime_out=dummy_out)
        assert est._out_format == "rotmat"


# ---------------------------------------------------------------------------
# 2. run() -- zero-size ROI
# ---------------------------------------------------------------------------

class TestRunEmptyROI:
    def test_zero_size_roi_returns_zeros(self):
        est = _make_estimator(out_shape=["batch", 6])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        result = est.run(frame, (50, 50, 50, 50))   # x2==x1 -> empty ROI
        assert result == {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    def test_out_of_bounds_bbox_clamped(self):
        est = _make_estimator(out_shape=["batch", 6])
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        # bbox extends beyond frame -- should clamp and still work (or return zeros)
        result = est.run(frame, (-10, -10, 200, 200))
        assert "pitch" in result


# ---------------------------------------------------------------------------
# 3. run() -- 6D format
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# 4. run() -- rotation-matrix format
# ---------------------------------------------------------------------------

class TestRunRotMat:
    def test_rotmat_path_returns_angles(self):
        R = np.eye(3, dtype=np.float32).reshape(1, 3, 3)
        est = _make_estimator(out_shape=["batch", 3, 3], runtime_out=R)
        est._out_format = "rotmat"

        frame = np.zeros((200, 200, 3), dtype=np.uint8)
        result = est.run(frame, (10, 10, 190, 190))
        assert set(result.keys()) == {"pitch", "yaw", "roll"}


# ---------------------------------------------------------------------------
# 5. Static methods
# ---------------------------------------------------------------------------

class TestStaticMethods:
    def test_6d_to_rot_produces_orthonormal_matrix(self):
        from dms.modules.head_pose import HeadPoseEstimator
        v = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float64)
        R = HeadPoseEstimator._6d_to_rot(v)
        assert R.shape == (3, 3)
        # Columns should be approximately unit vectors
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
        # Build a rotation matrix where sy < 1e-6 (gimbal lock)
        R = np.array([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ])
        # sy = sqrt(R[0,0]^2 + R[1,0]^2) = sqrt(0+0) = 0 -> singular
        pitch, yaw, roll = HeadPoseEstimator._rot_to_euler(R)
        assert isinstance(pitch, float)
        assert roll == 0.0   # singular branch sets roll=0

    def test_6d_to_rot_near_zero_norm_safe(self):
        """Very small vectors should not cause division-by-zero."""
        from dms.modules.head_pose import HeadPoseEstimator
        v = np.zeros(6, dtype=np.float64)
        R = HeadPoseEstimator._6d_to_rot(v)
        assert R.shape == (3, 3)
        assert not np.any(np.isnan(R))
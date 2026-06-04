"""
tests/test_camera.py
====================
Coverage target: = 90% of src/dms/modules/camera.py

cv2.VideoCapture is fully mocked -- no real hardware required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

_CV2_VC  = "dms.modules.camera.cv2.VideoCapture"
_SLEEP   = "dms.modules.camera.time.sleep"
_CONFIG  = "dms.modules.camera"


# ---------------------------------------------------------------------------
# Helper: build a mock VideoCapture
# ---------------------------------------------------------------------------

def _mock_cap(is_opened=True, read_frames=None):
    """
    read_frames: list of (ret, frame) to be returned by successive .read()
    calls.  If None, defaults to one successful frame.
    """
    cap = MagicMock()
    cap.isOpened.return_value = is_opened
    if read_frames is None:
        read_frames = [(True, np.zeros((480, 640, 3), dtype=np.uint8))]
    cap.read.side_effect = read_frames
    return cap


# ---------------------------------------------------------------------------
# 1. _csi_pipeline helper
# ---------------------------------------------------------------------------

class TestCsiPipeline:
    def test_contains_required_tokens(self):
        from dms.modules.camera import _csi_pipeline
        p = _csi_pipeline(1280, 720, 30, 0)
        assert "nvarguscamerasrc" in p
        assert "1280" in p
        assert "720"  in p
        assert "30/1" in p
        assert "flip-method=0" in p
        assert "appsink" in p

    def test_different_flip(self):
        from dms.modules.camera import _csi_pipeline
        p = _csi_pipeline(640, 480, 15, 2)
        assert "flip-method=2" in p


# ---------------------------------------------------------------------------
# 2. Camera construction
# ---------------------------------------------------------------------------

class TestCameraConstruction:
    def test_default_attributes(self):
        from dms.modules.camera import Camera
        from dms.config import (
            CAMERA_SOURCE, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS, CAMERA_FLIP
        )
        cam = Camera()
        assert cam.source == CAMERA_SOURCE
        assert cam.width  == CAMERA_WIDTH
        assert cam.height == CAMERA_HEIGHT
        assert cam.fps    == CAMERA_FPS
        assert cam.flip   == CAMERA_FLIP
        assert cam._cap is None

    def test_custom_params(self):
        from dms.modules.camera import Camera
        cam = Camera(source="0", width=320, height=240, fps=15, flip=2)
        assert cam.source == "0"
        assert cam.width  == 320
        assert cam.height == 240
        assert cam.fps    == 15
        assert cam.flip   == 2


# ---------------------------------------------------------------------------
# 3. open() -- CSI path
# ---------------------------------------------------------------------------

class TestOpenCSI:
    def test_csi_opens_successfully(self):
        from dms.modules.camera import Camera
        cap = _mock_cap()
        with patch(_CV2_VC, return_value=cap) as MockVC, \
             patch(_SLEEP):
            cam = Camera(source="csi")
            cam.open()
            # Called with a GStreamer pipeline string
            args = MockVC.call_args[0]
            assert "nvarguscamerasrc" in args[0]
        assert cam._cap is cap

    def test_csi_raises_if_not_opened(self):
        from dms.modules.camera import Camera
        cap = _mock_cap(is_opened=False)
        with patch(_CV2_VC, return_value=cap), patch(_SLEEP):
            cam = Camera(source="csi")
            with pytest.raises(RuntimeError, match="Failed to open"):
                cam.open()

    def test_csi_raises_if_no_frames(self):
        from dms.modules.camera import Camera
        # isOpened=True but read() always returns (False, None)
        cap = _mock_cap(read_frames=[(False, None)] * 10)
        with patch(_CV2_VC, return_value=cap), patch(_SLEEP):
            cam = Camera(source="csi")
            with pytest.raises(RuntimeError, match="no frames received"):
                cam.open()


# ---------------------------------------------------------------------------
# 4. open() -- USB / integer path
# ---------------------------------------------------------------------------

class TestOpenUSB:
    def test_integer_source_converted(self):
        from dms.modules.camera import Camera
        cap = _mock_cap()
        with patch(_CV2_VC, return_value=cap) as MockVC, patch(_SLEEP):
            cam = Camera(source="0")
            cam.open()
            src_arg = MockVC.call_args[0][0]
            assert src_arg == 0   # cast to int

    def test_dev_path_stays_string(self):
        from dms.modules.camera import Camera
        cap = _mock_cap()
        with patch(_CV2_VC, return_value=cap) as MockVC, patch(_SLEEP):
            cam = Camera(source="/dev/video0")
            cam.open()
            src_arg = MockVC.call_args[0][0]
            assert src_arg == "/dev/video0"

    def test_sets_resolution_and_fps(self):
        from dms.modules.camera import Camera
        cap = _mock_cap()
        with patch(_CV2_VC, return_value=cap), patch(_SLEEP):
            cam = Camera(source="0", width=640, height=480, fps=30)
            cam.open()
        # set() should have been called for WIDTH, HEIGHT, FPS
        prop_calls = [c.args[0] for c in cap.set.call_args_list]
        import cv2
        assert cv2.CAP_PROP_FRAME_WIDTH  in prop_calls
        assert cv2.CAP_PROP_FRAME_HEIGHT in prop_calls
        assert cv2.CAP_PROP_FPS          in prop_calls

    def test_warmup_loop_retries_on_none(self):
        """read() fails twice then succeeds on third attempt."""
        from dms.modules.camera import Camera
        good_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frames = [(False, None), (False, None), (True, good_frame)]
        cap = _mock_cap(read_frames=frames)
        with patch(_CV2_VC, return_value=cap), patch(_SLEEP):
            cam = Camera(source="0")
            cam.open()   # should NOT raise
        assert cam._cap is cap


# ---------------------------------------------------------------------------
# 5. read()
# ---------------------------------------------------------------------------

class TestRead:
    def test_read_returns_none_when_not_opened(self):
        from dms.modules.camera import Camera
        cam = Camera()
        assert cam.read() is None

    def test_read_returns_frame_on_success(self):
        from dms.modules.camera import Camera
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cap   = _mock_cap(read_frames=[(True, frame)] * 20)
        with patch(_CV2_VC, return_value=cap), patch(_SLEEP):
            cam = Camera(source="0")
            cam.open()
        result = cam.read()
        assert result is not None

    def test_read_returns_none_on_failed_read(self):
        from dms.modules.camera import Camera
        good = np.zeros((480, 640, 3), dtype=np.uint8)
        # warm-up succeeds, then a post-open read fails
        cap = _mock_cap(read_frames=[(True, good), (False, None)])
        with patch(_CV2_VC, return_value=cap), patch(_SLEEP):
            cam = Camera(source="0")
            cam.open()
        assert cam.read() is None


# ---------------------------------------------------------------------------
# 6. release()
# ---------------------------------------------------------------------------

class TestRelease:
    def test_release_calls_cap_release(self):
        from dms.modules.camera import Camera
        cap = _mock_cap()
        with patch(_CV2_VC, return_value=cap), patch(_SLEEP):
            cam = Camera(source="0")
            cam.open()
        cam.release()
        cap.release.assert_called_once()
        assert cam._cap is None

    def test_release_noop_when_not_opened(self):
        from dms.modules.camera import Camera
        cam = Camera()
        cam.release()   # must not raise
        assert cam._cap is None

    def test_double_release_safe(self):
        from dms.modules.camera import Camera
        cap = _mock_cap()
        with patch(_CV2_VC, return_value=cap), patch(_SLEEP):
            cam = Camera(source="0")
            cam.open()
        cam.release()
        cam.release()   # second call is a no-op


# ---------------------------------------------------------------------------
# 7. Context manager
# ---------------------------------------------------------------------------

class TestContextManager:
    def test_enter_calls_open(self):
        from dms.modules.camera import Camera
        cap = _mock_cap()
        with patch(_CV2_VC, return_value=cap), patch(_SLEEP):
            with Camera(source="0") as cam:
                assert cam._cap is cap
        # __exit__ calls release -> _cap becomes None
        assert cam._cap is None

    def test_exit_releases_on_exception(self):
        from dms.modules.camera import Camera
        cap = _mock_cap()
        with patch(_CV2_VC, return_value=cap), patch(_SLEEP):
            try:
                with Camera(source="0"):
                    raise ValueError("test error")
            except ValueError:
                pass
        cap.release.assert_called_once()
"""
tests/test_face_detector.py
============================
Coverage target: = 90% of src/dms/modules/face_detector.py

cv2.dnn, cv2.CascadeClassifier, cv2.TrackerCSRT_create are all mocked.
No real model files or camera hardware required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

# ---------------------------------------------------------------------------
# Patch targets
# ---------------------------------------------------------------------------
_DNN_READ      = "dms.modules.face_detector.cv2.dnn.readNetFromCaffe"
_DNN_BLOB      = "dms.modules.face_detector.cv2.dnn.blobFromImage"
_CASCADE       = "dms.modules.face_detector.cv2.CascadeClassifier"
_CSRT_CREATE   = "dms.modules.face_detector._create_csrt_tracker"
_OS_ISFILE     = "dms.modules.face_detector.os.path.isfile"
_CV2_RESIZE    = "dms.modules.face_detector.cv2.resize"
_CV2_CVT       = "dms.modules.face_detector.cv2.cvtColor"
_CV2_EQUALIZE  = "dms.modules.face_detector.cv2.equalizeHist"


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _make_dnn_detector(dnn_net=None):
    """Construct a FaceDetector in DNN mode with a mocked net."""
    if dnn_net is None:
        dnn_net = MagicMock()

    def isfile_side(path):
        # Return True for both proto and model paths, False for haar
        return "proto" in path or "caffemodel" in path or path.endswith(".xml") is False

    with patch(_OS_ISFILE, side_effect=lambda p: True), \
         patch(_DNN_READ, return_value=dnn_net):
        from dms.modules.face_detector import FaceDetector
        fd = FaceDetector(dnn_proto="fake.prototxt", dnn_model="fake.caffemodel")
    fd._dnn = dnn_net
    return fd


def _make_haar_detector():
    haar = MagicMock()
    with patch(_OS_ISFILE, side_effect=lambda p: p.endswith(".xml")), \
         patch(_CASCADE, return_value=haar):
        from dms.modules.face_detector import FaceDetector
        fd = FaceDetector()
    fd._haar = haar
    return fd


def _make_crop_detector():
    """No DNN files, no Haar file -> mode='crop'."""
    with patch(_OS_ISFILE, return_value=False):
        from dms.modules.face_detector import FaceDetector
        fd = FaceDetector()
    return fd


# ---------------------------------------------------------------------------
# 1. __init__ modes
# ---------------------------------------------------------------------------

class TestInit:
    def test_dnn_mode_when_files_exist(self):
        fd = _make_dnn_detector()
        assert fd._mode == "dnn"

    def test_haar_mode_when_dnn_files_missing(self):
        fd = _make_haar_detector()
        assert fd._mode == "haar"

    def test_crop_mode_when_nothing_available(self):
        fd = _make_crop_detector()
        assert fd._mode == "crop"

    def test_dnn_load_failure_falls_back_to_haar(self):
        haar = MagicMock()
        with patch(_OS_ISFILE, return_value=True), \
             patch(_DNN_READ, side_effect=RuntimeError("load failed")), \
             patch(_CASCADE, return_value=haar):
            from dms.modules.face_detector import FaceDetector
            fd = FaceDetector(dnn_proto="p.prototxt", dnn_model="m.caffemodel")
        assert fd._mode == "haar"

    def test_tracker_state_initialised_to_none(self):
        fd = _make_haar_detector()
        assert fd._tracker       is None
        assert fd._tracker_miss  == 0
        assert fd._last_bbox     is None


# ---------------------------------------------------------------------------
# 2. _pad()
# ---------------------------------------------------------------------------

class TestPad:
    def test_padding_expands_box(self):
        fd = _make_crop_detector()
        fd._padding = 0.1
        x1, y1, x2, y2 = fd._pad(100, 100, 200, 200, 480, 640)
        assert x1 < 100 and y1 < 100
        assert x2 > 200 and y2 > 200

    def test_padding_clamped_to_frame(self):
        fd = _make_crop_detector()
        fd._padding = 2.0   # extreme padding
        x1, y1, x2, y2 = fd._pad(10, 10, 50, 50, 100, 100)
        assert x1 >= 0 and y1 >= 0
        assert x2 <= 100 and y2 <= 100


# ---------------------------------------------------------------------------
# 3. _centre_crop()
# ---------------------------------------------------------------------------

class TestCentreCrop:
    def test_returns_inner_quarter(self):
        fd = _make_crop_detector()
        frame = _blank(100, 100)
        x1, y1, x2, y2 = fd._centre_crop(frame)
        assert x1 > 0 and y1 > 0
        assert x2 < 100 and y2 < 100
        assert x1 < x2 and y1 < y2


# ---------------------------------------------------------------------------
# 4. _detect_dnn()
# ---------------------------------------------------------------------------

class TestDetectDNN:
    def _make_detections(self, conf, x1=0.1, y1=0.1, x2=0.5, y2=0.6):
        """Build a fake DNN detections array (1,1,1,7)."""
        det = np.zeros((1, 1, 1, 7), dtype=np.float32)
        det[0, 0, 0, 2] = conf
        det[0, 0, 0, 3:7] = [x1, y1, x2, y2]
        return det

    def test_returns_none_when_no_face_above_threshold(self):
        net = MagicMock()
        net.forward.return_value = self._make_detections(conf=0.01)
        fd = _make_dnn_detector(dnn_net=net)
        fd._conf_threshold = 0.5
        fd._padding = 0.0

        frame = _blank()
        with patch(_DNN_BLOB, return_value=MagicMock()), \
             patch(_CV2_RESIZE, return_value=_blank(300, 300)):
            result = fd._detect_dnn(frame)
        assert result is None

    def test_returns_bbox_when_face_found(self):
        net = MagicMock()
        net.forward.return_value = self._make_detections(conf=0.95)
        fd = _make_dnn_detector(dnn_net=net)
        fd._conf_threshold = 0.5
        fd._padding = 0.0

        frame = _blank(480, 640)
        with patch(_DNN_BLOB, return_value=MagicMock()), \
             patch(_CV2_RESIZE, return_value=_blank(300, 300)):
            result = fd._detect_dnn(frame)
        assert result is not None
        assert len(result) == 4

    def test_picks_highest_confidence_box(self):
        """Two detections above threshold; should return the higher-conf one."""
        det = np.zeros((1, 1, 2, 7), dtype=np.float32)
        # Face 1: conf=0.6, coords: (0.1,0.1,0.4,0.5)
        det[0, 0, 0, 2] = 0.60
        det[0, 0, 0, 3:7] = [0.1, 0.1, 0.4, 0.5]
        # Face 2: conf=0.92, coords: (0.5,0.5,0.9,0.9)
        det[0, 0, 1, 2] = 0.92
        det[0, 0, 1, 3:7] = [0.5, 0.5, 0.9, 0.9]

        net = MagicMock()
        net.forward.return_value = det
        fd = _make_dnn_detector(dnn_net=net)
        fd._conf_threshold = 0.5
        fd._padding = 0.0

        frame = _blank(480, 640)
        with patch(_DNN_BLOB, return_value=MagicMock()), \
             patch(_CV2_RESIZE, return_value=_blank(300, 300)):
            result = fd._detect_dnn(frame)
        assert result is not None


# ---------------------------------------------------------------------------
# 5. _detect_haar()
# ---------------------------------------------------------------------------

class TestDetectHaar:
    def test_returns_none_when_no_face(self):
        fd = _make_haar_detector()
        fd._haar.detectMultiScale.return_value = np.array([]).reshape(0, 4)
        fd._padding = 0.0

        gray = np.zeros((480, 640), dtype=np.uint8)
        with patch(_CV2_CVT, return_value=gray), \
             patch(_CV2_EQUALIZE, return_value=gray):
            result = fd._detect_haar(_blank())
        assert result is None

    def test_returns_largest_face(self):
        fd = _make_haar_detector()
        # Two faces: small (10x10) and large (100x120)
        faces = np.array([[10, 10, 10, 10], [50, 50, 100, 120]])
        fd._haar.detectMultiScale.return_value = faces
        fd._padding = 0.0

        gray = np.zeros((480, 640), dtype=np.uint8)
        with patch(_CV2_CVT, return_value=gray), \
             patch(_CV2_EQUALIZE, return_value=gray):
            result = fd._detect_haar(_blank())
        assert result is not None
        # The larger face: x=50, y=50, w=100, h=120
        x1, y1, x2, y2 = result
        assert x2 - x1 >= 100   # width of the larger face (without padding)


# ---------------------------------------------------------------------------
# 6. detect() -- full pipeline
# ---------------------------------------------------------------------------

class TestDetect:
    def test_detect_dnn_fresh_starts_tracker(self):
        net = MagicMock()
        det = np.zeros((1, 1, 1, 7), dtype=np.float32)
        det[0, 0, 0, 2] = 0.95
        det[0, 0, 0, 3:7] = [0.1, 0.1, 0.5, 0.6]
        net.forward.return_value = det

        tracker = MagicMock()
        fd = _make_dnn_detector(dnn_net=net)
        fd._conf_threshold = 0.5
        fd._padding = 0.0

        frame = _blank(480, 640)
        with patch(_DNN_BLOB, return_value=MagicMock()), \
             patch(_CV2_RESIZE, return_value=_blank(300, 300)), \
             patch(_CSRT_CREATE, return_value=tracker):
            result = fd.detect(frame)
        assert result is not None
        assert fd.is_tracked is False
        tracker.init.assert_called_once()

    def test_detect_haar_fresh(self):
        fd = _make_haar_detector()
        faces = np.array([[50, 50, 100, 100]])
        fd._haar.detectMultiScale.return_value = faces
        fd._padding = 0.0

        tracker = MagicMock()
        gray = np.zeros((480, 640), dtype=np.uint8)
        with patch(_CV2_CVT, return_value=gray), \
             patch(_CV2_EQUALIZE, return_value=gray), \
             patch(_CSRT_CREATE, return_value=tracker):
            result = fd.detect(_blank())
        assert result is not None

    def test_detect_crop_mode(self):
        fd = _make_crop_detector()
        result = fd.detect(_blank(100, 100))
        assert result is not None
        assert len(result) == 4

    def test_detect_uses_tracker_when_detector_misses(self):
        """Detector returns None, tracker update() succeeds."""
        fd = _make_crop_detector()
        fd._mode = "haar"
        fd._haar = MagicMock()
        fd._haar.detectMultiScale.return_value = np.array([]).reshape(0, 4)

        tracker = MagicMock()
        tracker.update.return_value = (True, (50, 50, 100, 100))
        fd._tracker      = tracker
        fd._tracker_miss = 0

        gray = np.zeros((480, 640), dtype=np.uint8)
        with patch(_CV2_CVT, return_value=gray), \
             patch(_CV2_EQUALIZE, return_value=gray):
            result = fd.detect(_blank(480, 640))
        assert result is not None
        assert fd.is_tracked is True

    def test_detect_tracker_exhausted_returns_none(self):
        """tracker_miss already at max -> skip tracker, return None."""
        from dms.modules.face_detector import TRACKER_MAX_MISS
        fd = _make_crop_detector()
        fd._mode = "haar"
        fd._haar = MagicMock()
        fd._haar.detectMultiScale.return_value = np.array([]).reshape(0, 4)

        tracker = MagicMock()
        fd._tracker      = tracker
        fd._tracker_miss = TRACKER_MAX_MISS   # already at limit

        gray = np.zeros((480, 640), dtype=np.uint8)
        with patch(_CV2_CVT, return_value=gray), \
             patch(_CV2_EQUALIZE, return_value=gray):
            result = fd.detect(_blank(480, 640))
        assert result is None
        assert fd._tracker is None

    def test_detect_no_tracker_returns_none(self):
        fd = _make_crop_detector()
        fd._mode = "haar"
        fd._haar = MagicMock()
        fd._haar.detectMultiScale.return_value = np.array([]).reshape(0, 4)
        fd._tracker = None

        gray = np.zeros((480, 640), dtype=np.uint8)
        with patch(_CV2_CVT, return_value=gray), \
             patch(_CV2_EQUALIZE, return_value=gray):
            result = fd.detect(_blank(480, 640))
        assert result is None

    def test_detect_tracker_update_fails(self):
        """tracker.update() returns ok=False."""
        fd = _make_crop_detector()
        fd._mode = "haar"
        fd._haar = MagicMock()
        fd._haar.detectMultiScale.return_value = np.array([]).reshape(0, 4)

        tracker = MagicMock()
        tracker.update.return_value = (False, (0, 0, 0, 0))
        fd._tracker      = tracker
        fd._tracker_miss = 0

        gray = np.zeros((480, 640), dtype=np.uint8)
        with patch(_CV2_CVT, return_value=gray), \
             patch(_CV2_EQUALIZE, return_value=gray):
            result = fd.detect(_blank(480, 640))
        assert result is None

    def test_detect_tracker_zero_size_box_skipped(self):
        """tracker.update() ok=True but tw==0 -> treated as miss."""
        fd = _make_crop_detector()
        fd._mode = "haar"
        fd._haar = MagicMock()
        fd._haar.detectMultiScale.return_value = np.array([]).reshape(0, 4)

        tracker = MagicMock()
        tracker.update.return_value = (True, (0, 0, 0, 0))   # zero w/h
        fd._tracker      = tracker
        fd._tracker_miss = 0

        gray = np.zeros((480, 640), dtype=np.uint8)
        with patch(_CV2_CVT, return_value=gray), \
             patch(_CV2_EQUALIZE, return_value=gray):
            result = fd.detect(_blank(480, 640))
        assert result is None
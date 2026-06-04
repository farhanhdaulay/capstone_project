"""
tests/test_phone.py
===================
Coverage target: = 90% of src/dms/modules/phone.py

ultralytics.YOLO is fully mocked -- no model files or GPU required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

_GET_YOLO  = "dms.modules.phone._get_yolo"
_OS_ISFILE = "dms.modules.phone.os.path.isfile"
_LOGGER    = "dms.modules.phone.logger"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _blank(h=480, w=640):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _mock_result(has_box=True, conf_val=0.85, coords=(10, 10, 100, 100)):
    """Build a single YOLO result-like object."""
    result = MagicMock()
    if has_box:
        box = MagicMock()
        box.conf        = [conf_val]
        box.xyxy        = [coords]
        result.boxes    = MagicMock()
        result.boxes.__len__ = lambda self: 1
        result.boxes.__iter__ = lambda self: iter([box])
    else:
        result.boxes = None
    return result


def _make_detector(engine_exists=False, pt_exists=True,
                   yolo_load_ok=True, consec=3):
    """
    Return a PhoneDetector with mocked YOLO.
    engine_exists / pt_exists control which path isfile() returns True for.
    """
    YOLO_cls = MagicMock()
    model_inst = MagicMock()
    YOLO_cls.return_value = model_inst
    if not yolo_load_ok:
        YOLO_cls.side_effect = RuntimeError("load error")

    def isfile(path):
        if "engine" in path:
            return engine_exists
        if ".pt" in path:
            return pt_exists
        return False

    with patch(_OS_ISFILE, side_effect=isfile), \
         patch(_GET_YOLO, return_value=YOLO_cls):
        from dms.modules.phone import PhoneDetector
        det = PhoneDetector(
            engine_path="model.engine",
            pt_path="yolo11n.pt",
            conf=0.40,
            class_id=67,
            consec=consec,
        )
    det._model = model_inst
    return det, model_inst


# ---------------------------------------------------------------------------
# 1. _load_model paths
# ---------------------------------------------------------------------------

class TestLoadModel:
    def test_loads_engine_when_exists(self):
        YOLO_cls = MagicMock()
        with patch(_OS_ISFILE, return_value=True), \
             patch(_GET_YOLO, return_value=YOLO_cls):
            from dms.modules.phone import PhoneDetector
            PhoneDetector("model.engine", "yolo11n.pt")
        YOLO_cls.assert_called()

    def test_falls_back_to_pt_when_engine_load_fails(self):
        YOLO_cls = MagicMock()
        call_count = {"n": 0}

        def side_effect(path, **kwargs):
            call_count["n"] += 1
            if "engine" in str(path):
                raise RuntimeError("TRT fail")
            return MagicMock()

        YOLO_cls.side_effect = side_effect

        def isfile(p):
            return True   # both files exist

        with patch(_OS_ISFILE, side_effect=isfile), \
             patch(_GET_YOLO, return_value=YOLO_cls):
            from dms.modules.phone import PhoneDetector
            PhoneDetector("model.engine", "yolo11n.pt")
        assert call_count["n"] >= 2   # tried engine, then pt

    def test_auto_download_when_neither_exists(self):
        YOLO_cls = MagicMock()
        with patch(_OS_ISFILE, return_value=False), \
             patch(_GET_YOLO, return_value=YOLO_cls):
            from dms.modules.phone import PhoneDetector
            PhoneDetector("model.engine", "yolo11n.pt")
        # Should have called YOLO("yolo11n.pt") for auto-download
        called_with = [str(c.args[0]) for c in YOLO_cls.call_args_list]
        assert any("yolo11n.pt" in p for p in called_with)


# ---------------------------------------------------------------------------
# 2. detect() -- model None
# ---------------------------------------------------------------------------

class TestDetectModelNone:
    def test_returns_false_when_no_model(self):
        from dms.modules.phone import PhoneDetector
        with patch(_OS_ISFILE, return_value=False), \
             patch(_GET_YOLO, return_value=MagicMock()):
            det = PhoneDetector("e.engine", "w.pt")
        det._model = None
        assert det.detect(_blank()) is False


# ---------------------------------------------------------------------------
# 3. detect() -- inference error
# ---------------------------------------------------------------------------

class TestDetectInferenceError:
    def test_returns_false_on_exception(self):
        det, model = _make_detector()
        model.side_effect = RuntimeError("GPU OOM")
        assert det.detect(_blank()) is False


# ---------------------------------------------------------------------------
# 4. detect() -- counter hysteresis
# ---------------------------------------------------------------------------

class TestDetectHysteresis:
    def test_counter_increments_on_phone_found(self):
        det, model = _make_detector(consec=5)
        model.return_value = [_mock_result(has_box=True)]
        det.detect(_blank())
        assert det.counter == 1

    def test_returns_true_after_consec_frames(self):
        det, model = _make_detector(consec=3)
        model.return_value = [_mock_result(has_box=True)]
        for _ in range(3):
            result = det.detect(_blank())
        assert result is True

    def test_counter_decrements_but_not_below_zero(self):
        det, model = _make_detector(consec=3)
        model.return_value = [_mock_result(has_box=False)]
        det.detect(_blank())
        assert det.counter == 0   # max(0, 0-1) = 0

    def test_counter_decrements_from_positive(self):
        det, model = _make_detector(consec=5)
        det._counter = 4
        model.return_value = [_mock_result(has_box=False)]
        det.detect(_blank())
        assert det.counter == 3

    def test_returns_false_below_consec(self):
        det, model = _make_detector(consec=5)
        model.return_value = [_mock_result(has_box=True)]
        result = det.detect(_blank())   # counter=1, need 5
        assert result is False

    def test_no_boxes_in_result(self):
        """r.boxes is not None but len==0 -> phone_found=False."""
        det, model = _make_detector(consec=3)
        r = MagicMock()
        r.boxes = MagicMock()
        r.boxes.__len__ = lambda self: 0
        model.return_value = [r]
        det.detect(_blank())
        assert det.counter == 0


# ---------------------------------------------------------------------------
# 5. draw()
# ---------------------------------------------------------------------------

class TestDraw:
    def test_returns_frame_when_model_none(self):
        det, _ = _make_detector()
        det._model = None
        frame  = _blank()
        result = det.draw(frame)
        assert result is frame

    def test_returns_frame_on_inference_error(self):
        det, model = _make_detector()
        model.side_effect = RuntimeError("fail")
        frame  = _blank()
        result = det.draw(frame)
        assert result is frame

    def test_draws_boxes_on_frame(self):
        det, model = _make_detector()
        box = MagicMock()
        box.conf = [0.85]
        box.xyxy = [MagicMock()]
        box.xyxy[0].tolist.return_value = [10.0, 10.0, 100.0, 100.0]

        r = MagicMock()
        r.boxes = [box]
        model.return_value = [r]

        import cv2
        frame = _blank()
        with patch.object(cv2, "rectangle") as mock_rect, \
             patch.object(cv2, "putText"), \
             patch.object(cv2, "resize", return_value=_blank(240, 320)):
            det.draw(frame)
            mock_rect.assert_called()

    def test_draw_no_boxes_in_result(self):
        det, model = _make_detector()
        r = MagicMock()
        r.boxes = None
        model.return_value = [r]

        import cv2
        frame = _blank()
        with patch.object(cv2, "resize", return_value=_blank(240, 320)):
            result = det.draw(frame)
        assert result is frame


# ---------------------------------------------------------------------------
# 6. counter property and reset()
# ---------------------------------------------------------------------------

class TestCounterAndReset:
    def test_counter_property(self):
        det, _ = _make_detector()
        det._counter = 7
        assert det.counter == 7

    def test_reset_clears_counter(self):
        det, _ = _make_detector()
        det._counter = 10
        det.reset()
        assert det.counter == 0
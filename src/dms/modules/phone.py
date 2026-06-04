"""
phone.py - Cell-phone distraction detector.

Uses a TensorRT-compiled YOLOv11n engine (yolo11n.engine) on Jetson
for maximum inference speed.  Falls back to the PyTorch .pt weights
when the engine is absent (development / CPU machines).

Provides : PhoneDetector - call .detect(frame) ? bool
"""

from __future__ import annotations

import logging
import os

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Lazy import so the module loads even without ultralytics installed
_YOLO_cls = None

def _get_yolo():
    global _YOLO_cls
    if _YOLO_cls is None:
        from ultralytics import YOLO  # type: ignore
        _YOLO_cls = YOLO
    return _YOLO_cls


class PhoneDetector:
    """
    Detects whether a mobile phone is visible in the frame.

    Parameters
    ----------
    engine_path : Path to the TensorRT .engine file (preferred).
    pt_path     : Fallback PyTorch .pt weights.
    conf        : Confidence threshold (0-1).
    class_id    : COCO class index for 'cell phone' (default 67).
    consec      : Consecutive frames phone must appear before flagging.
    """

    def __init__(
        self,
        engine_path: str,
        pt_path: str,
        conf: float = 0.40,
        class_id: int = 67,
        consec: int = 5,
    ) -> None:
        self._conf      = conf
        self._class_id  = class_id
        self._consec    = consec
        self._counter   = 0
        self._model     = None

        self._load_model(engine_path, pt_path)

    # -- Model loading ---------------------------------------------------------
    def _load_model(self, engine_path: str, pt_path: str) -> None:
        YOLO = _get_yolo()

        # Prefer TensorRT engine (fastest on Jetson)
        if os.path.isfile(engine_path):
            try:
                self._model = YOLO(engine_path, task="detect")
                logger.info("PhoneDetector loaded TensorRT engine: %s", engine_path)
                return
            except Exception as exc:
                logger.warning("Engine load failed (%s) - trying .pt", exc)

        # Fallback: PyTorch weights (auto-downloaded if missing)
        if os.path.isfile(pt_path):
            self._model = YOLO(pt_path)
            logger.info("PhoneDetector loaded PT weights: %s", pt_path)
        else:
            logger.warning(
                "Neither engine (%s) nor PT (%s) found - "
                "attempting to download yolo11n.pt via ultralytics.",
                engine_path, pt_path,
            )
            self._model = YOLO("yolo11n.pt")   # ultralytics auto-download
            logger.info("PhoneDetector downloaded yolo11n.pt.")

    # -- Inference -------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> bool:
        """
        Run inference on *frame* and return True when a phone is detected
        for at least `consec` consecutive frames.

        Parameters
        ----------
        frame : BGR image (H?W?3 uint8) from OpenCV.

        Returns
        -------
        bool - True = phone detected (sustained), False otherwise.
        """
        if self._model is None:
            return False

        try:
            results = self._model(
                frame,
                conf=self._conf,
                classes=[self._class_id],
                verbose=False,
            )
        except Exception as exc:
            logger.error("PhoneDetector inference error: %s", exc)
            return False

        phone_found = False
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                phone_found = True
                break

        # Hysteresis: require `consec` consecutive positive frames
        if phone_found:
            self._counter += 1
        else:
            self._counter = max(0, self._counter - 1)

        return self._counter >= self._consec

    def draw(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw detection boxes on *frame* (in-place) and return it.
        Useful for debugging / visualisation.
        """
        if self._model is None:
            return frame

        try:
            ph, pw = frame.shape[:2]
            phone_frame = cv2.resize(frame, (320, 240))
            self._scale_x = pw / 320
            self._scale_y = ph / 240
            results = self._model(
                phone_frame,  
                conf=self._conf,
                classes=[self._class_id],
                verbose=False,
            )
            for r in results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    bx1, by1, bx2, by2 = box.xyxy[0].tolist()
                    sx = getattr(self, '_scale_x', 1.0)
                    sy = getattr(self, '_scale_y', 1.0)
                    x1 = int(bx1 * sx)
                    y1 = int(by1 * sy)
                    x2 = int(bx2 * sx)
                    y2 = int(by2 * sy)
                    conf_val = float(box.conf[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"Phone {conf_val:.2f}",
                        (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )
        except Exception as exc:
            logger.error("PhoneDetector draw error: %s", exc)

        return frame

    @property
    def counter(self) -> int:
        """Current consecutive-frame counter (diagnostic)."""
        return self._counter

    def reset(self) -> None:
        """Reset the consecutive counter (e.g. after alert is handled)."""
        self._counter = 0
# src/dms/modules/face_detector.py
"""
Face detector module.
Primary:  OpenCV DNN with a lightweight SSD face model (res10_300x300).
Fallback: Haar cascade (always available, no extra model file needed).

When the detector misses (e.g. turned head), a CSRT tracker keeps the
last known bounding box alive for up to TRACKER_MAX_MISS frames so that
PFLD and head-pose can keep running through brief detection gaps.
"""

import cv2
import numpy as np
import os
from dms.config import (
    FACE_CONF_THRESHOLD, FACE_SCALE_FACTOR,
    FACE_MIN_NEIGHBOURS, FACE_MIN_SIZE, FACE_PADDING
)
def _create_csrt_tracker():
    """Thin wrapper so tests can patch tracker creation easily."""
    try:
        return cv2.legacy.TrackerCSRT_create()
    except AttributeError:
        try:
            return cv2.TrackerCSRT_create()
        except AttributeError:
            return None   # no tracker available on this build

TRACKER_MAX_MISS = 15   # frames to trust tracker after detector loses face


class FaceDetector:
    def __init__(
        self,
        model_path:     str   = None,
        dnn_proto:      str   = None,
        dnn_model:      str   = None,
        conf_threshold: float = None,
        scale_factor:   float = None,
        min_neighbours: int   = None,
        min_size:       tuple = None,
        padding:        float = None,
    ):
        self._conf_threshold = conf_threshold  if conf_threshold  is not None else FACE_CONF_THRESHOLD
        self._scale_factor   = scale_factor    if scale_factor    is not None else FACE_SCALE_FACTOR
        self._min_neighbours = min_neighbours  if min_neighbours  is not None else FACE_MIN_NEIGHBOURS
        self._min_size       = min_size        if min_size        is not None else FACE_MIN_SIZE
        self._padding        = padding         if padding         is not None else FACE_PADDING

        self._dnn  = None
        self._haar = None
        self._mode = "none"

        # Tracker state
        self._tracker      = None
        self._tracker_miss = 0
        self._last_bbox    = None   # last confirmed (x1,y1,x2,y2)

        # --- Try DNN first ---------------------------------------------------
        if dnn_proto and dnn_model and os.path.isfile(dnn_proto) and os.path.isfile(dnn_model):
            try:
                self._dnn  = cv2.dnn.readNetFromCaffe(dnn_proto, dnn_model)
                self._mode = "dnn"
                print("[FaceDetector] Using DNN (res10_300x300)")
                return
            except Exception as e:
                print(f"[FaceDetector] DNN load failed: {e} -- falling back to Haar")

        # --- Haar cascade fallback -------------------------------------------
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        if os.path.isfile(haar_path):
            self._haar = cv2.CascadeClassifier(haar_path)
            self._mode = "haar"
            print("[FaceDetector] Using Haar cascade (built-in)")
        else:
            print("[FaceDetector] WARN: no face detector available -- using centre crop")
            self._mode = "crop"

    # -- public API ------------------------------------------------------------

    def detect(self, frame: np.ndarray):
        """
        Detect the largest face.  Falls back to CSRT tracker when the
        detector misses (e.g. head turned sideways).

        Returns: (x1, y1, x2, y2) or None
        is_tracked attribute is set True when the result comes from the
        tracker rather than a fresh detection (main.py uses this to choose
        the box colour).
        """
        self.is_tracked = False   # reset each call

        # 1 -- Run the detector
        if self._mode == "dnn":
            raw = self._detect_dnn(frame)
        elif self._mode == "haar":
            raw = self._detect_haar(frame)
        else:
            raw = self._centre_crop(frame)

        # 2 -- Fresh detection: restart tracker on the new bbox
        if raw is not None:
            self._tracker_miss = 0
            self._last_bbox    = raw
            # Reinitialise tracker so it tracks the freshly confirmed face
            # Just use:
            self._tracker = _create_csrt_tracker()
            if self._tracker is not None:
                x1, y1, x2, y2 = raw
                self._tracker.init(frame, (x1, y1, x2 - x1, y2 - y1))
            return raw

        # 3 -- Detector missed -- try the tracker
        if self._tracker is not None and self._tracker_miss < TRACKER_MAX_MISS:
            ok, box = self._tracker.update(frame)
            self._tracker_miss += 1
            if ok:
                tx, ty, tw, th = [int(v) for v in box]
                # Clamp tracker output to frame bounds
                fh, fw = frame.shape[:2]
                tx = max(0, tx) 
                ty = max(0, ty)
                tw = min(tw, fw - tx)
                th = min(th, fh - ty)
                if tw > 0 and th > 0:
                    bbox = (tx, ty, tx + tw, ty + th)
                    self._last_bbox = bbox
                    self.is_tracked = True
                    return bbox

        # 4 -- Both detector and tracker failed
        self._tracker      = None
        self._tracker_miss = 0
        return None

    # -- internals -------------------------------------------------------------

    def _pad(self, x1, y1, x2, y2, h, w):
        pw = int((x2 - x1) * self._padding)
        ph = int((y2 - y1) * self._padding)
        return (max(0, x1 - pw), max(0, y1 - ph),
                min(w, x2 + pw), min(h, y2 + ph))

    def _detect_dnn(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 1.0,
            (300, 300), (104.0, 177.0, 123.0)
        )
        self._dnn.setInput(blob)
        detections = self._dnn.forward()

        best_conf, best_box = 0.0, None
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf < self._conf_threshold:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            x1, y1, x2, y2 = box.astype(int)
            if conf > best_conf:
                best_conf, best_box = conf, (x1, y1, x2, y2)

        if best_box is None:
            return None
        return self._pad(*best_box, h, w)

    def _detect_haar(self, frame):
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # equalizeHist improves detection in dim/uneven lighting
        gray  = cv2.equalizeHist(gray)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor  = self._scale_factor,
            minNeighbors = self._min_neighbours,
            minSize      = self._min_size,
        )
        if len(faces) == 0:
            return None

        h, w = frame.shape[:2]
        x, y, fw, fh = max(faces, key=lambda r: r[2] * r[3])
        return self._pad(x, y, x + fw, y + fh, h, w)

    def _centre_crop(self, frame):
        h, w = frame.shape[:2]
        m    = 0.15
        return (int(w * m), int(h * m), int(w * (1 - m)), int(h * (1 - m)))
# src/dms/modules/head_pose.py
"""
Head pose estimation using 6DRepNet360.

Handles two possible ONNX export formats:
  (1, 6)   -- 6D rotation representation (most common export)
  (1, 3, 3)-- rotation matrix (less common)

The correct format is detected automatically at load time.
ROI is clamped to frame bounds before slicing.
"""

import cv2
import numpy as np
import onnxruntime as ort
from dms.config import HEAD_POSE_MODEL


class HeadPoseEstimator:
    """
    Estimates head orientation from a face crop.

    Usage:
        estimator = HeadPoseEstimator()
        result    = estimator.run(frame, face_bbox)
        # result["pitch"], result["yaw"], result["roll"]  (degrees)
    """

    INPUT_SIZE = (224, 224)
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, model_path=HEAD_POSE_MODEL):
        providers     = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.sess     = ort.InferenceSession(model_path, providers=providers)
        self.in_name  = self.sess.get_inputs()[0].name
        self.out_name = self.sess.get_outputs()[0].name

        # Probe output shape so we know which decoder to use
        out_shape = self.sess.get_outputs()[0].shape
        print("[HeadPose] Loaded: {}".format(model_path))
        print("           provider = {}".format(self.sess.get_providers()[0]))
        print("           input    = {}".format(self.sess.get_inputs()[0].shape))
        print("           output   = {}".format(out_shape))

        # Collect only the integer dims (skip dynamic 'batch_size' strings)
        static = [d for d in out_shape if isinstance(d, int)]

        if len(static) == 1 and static[0] == 6:
            self._out_format = "6d"
            print("[HeadPose] Format: 6D rotation representation -> will convert to R")
        elif len(static) == 2 and static == [3, 3]:
            self._out_format = "rotmat"
            print("[HeadPose] Format: 3x3 rotation matrix")
        else:
            # Fall back: run a dummy pass and check runtime shape
            dummy = np.zeros((1, 3, self.INPUT_SIZE[0], self.INPUT_SIZE[1]),
                             dtype=np.float32)
            out = self.sess.run([self.out_name], {self.in_name: dummy})[0]
            print("[HeadPose] Runtime output shape: {} -- detecting format".format(out.shape))
            if out.shape[-1] == 6:
                self._out_format = "6d"
            else:
                self._out_format = "rotmat"
            print("[HeadPose] Resolved format: {}".format(self._out_format))

    # -- public API ------------------------------------------------------------

    def run(self, frame, face_bbox):
        """
        Args:
            frame:     full BGR frame (H x W x 3)
            face_bbox: (x1, y1, x2, y2) pixel coords

        Returns dict:
            pitch  float  (positive = looking down)
            yaw    float  (positive = turning right)
            roll   float  (positive = tilting right)
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
            return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

        # Preprocess
        img = cv2.resize(roi, self.INPUT_SIZE)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - self._MEAN) / self._STD
        inp = img.transpose(2, 0, 1)[np.newaxis, :]  # (1, 3, 224, 224)

        # Inference
        out = self.sess.run([self.out_name], {self.in_name: inp})[0]

        # Decode to rotation matrix
        if self._out_format == "6d":
            R = self._6d_to_rot(out.reshape(6))
        else:
            R = out.reshape(3, 3)

        pitch, yaw, roll = self._rot_to_euler(R)
        return {
            "pitch": round(float(pitch), 1),
            "yaw":   round(float(yaw),   1),
            "roll":  round(float(roll),  1),
        }

    # -- internals -------------------------------------------------------------

    @staticmethod
    def _6d_to_rot(v):
        """
        Convert 6D rotation representation to a 3x3 rotation matrix.
        Zhou et al. "On the Continuity of Rotation Representations" (CVPR 2019).

        v: array of shape (6,) -- first two column vectors of R, not yet orthonormal
        """
        a1 = v[0:3].astype(np.float64)
        a2 = v[3:6].astype(np.float64)

        b1 = a1 / (np.linalg.norm(a1) + 1e-8)
        b2 = a2 - np.dot(b1, a2) * b1
        b2 = b2 / (np.linalg.norm(b2) + 1e-8)
        b3 = np.cross(b1, b2)

        return np.stack([b1, b2, b3], axis=1)  # (3, 3) columns are basis vectors

    @staticmethod
    def _rot_to_euler(R):
        """
        Convert a 3x3 rotation matrix to Euler angles in degrees.
        Convention: R = Rz(roll) @ Ry(yaw) @ Rx(pitch)
        """
        sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
        singular = sy < 1e-6

        if not singular:
            pitch = np.degrees(np.arctan2( R[2, 1], R[2, 2]))
            yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
            roll  = np.degrees(np.arctan2( R[1, 0], R[0, 0]))
        else:
            pitch = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
            yaw   = np.degrees(np.arctan2(-R[2, 0], sy))
            roll  = 0.0

        return float(pitch), float(yaw), float(roll)
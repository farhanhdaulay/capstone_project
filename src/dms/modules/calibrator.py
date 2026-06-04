"""
calibrator.py -- Per-session EAR/MAR baseline calibration.
"""
from __future__ import annotations
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)


class EARCalibrator:
    """
    Collect EAR samples for `duration_s` seconds then lock thresholds.
    min_samples is intentionally low (5) so it works even at 2-3 FPS.
    """

    def __init__(
        self,
        duration_s:   float = 4.0,
        drowsy_ratio: float = 0.75,
        fallback_ear: float = 0.20,
        fallback_mar: float = 0.55,
        min_samples:  int   = 5,       # was 30 -- 5 is enough for a reliable mean
    ) -> None:
        self._duration_s   = duration_s
        self._drowsy_ratio = drowsy_ratio
        self._fallback_ear = fallback_ear
        self._fallback_mar = fallback_mar
        self._min_samples  = min_samples

        self._ear_samples: list[float] = []
        self._start: Optional[float]   = None
        self._done  = False

        self._ear_threshold: float = fallback_ear
        self._mar_threshold: float = fallback_mar

        logger.info(
            "EARCalibrator: will collect %.1f s of baseline "
            "(drowsy_ratio=%.2f, fallback_ear=%.2f, min_samples=%d)",
            duration_s, drowsy_ratio, fallback_ear, min_samples,
        )

    def update(self, ear: float, mar: float) -> None:
        if self._done:
            return
        if ear <= 0.0:
            return

        if self._start is None:
            self._start = time.monotonic()
            logger.info("EARCalibrator: baseline collection started -- keep eyes open.")

        self._ear_samples.append(ear)

        elapsed = time.monotonic() - self._start
        if elapsed >= self._duration_s:
            self._finalise()

    def _finalise(self) -> None:
        n = len(self._ear_samples)
        if n < self._min_samples:
            logger.warning(
                "EARCalibrator: only %d samples (need %d) -- using fallback EAR=%.2f",
                n, self._min_samples, self._fallback_ear,
            )
            self._ear_threshold = self._fallback_ear
        else:
            baseline = sum(self._ear_samples) / n
            self._ear_threshold = round(baseline * self._drowsy_ratio, 4)
            logger.info(
                "EARCalibrator done -- %d samples, baseline=%.4f, threshold=%.4f",
                n, baseline, self._ear_threshold,
            )

        self._done = True

    @property
    def done(self) -> bool:
        return self._done

    @property
    def ear_threshold(self) -> float:
        return self._ear_threshold

    @property
    def mar_threshold(self) -> float:
        return self._mar_threshold

    @property
    def progress(self) -> float:
        if self._done:
            return 1.0
        if self._start is None:
            return 0.0
        return min((time.monotonic() - self._start) / self._duration_s, 1.0)
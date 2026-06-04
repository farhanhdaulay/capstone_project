"""
tests/test_calibrator.py
========================
Coverage target: = 90% of src/dms/modules/calibrator.py

All time.monotonic() calls are patched so tests run instantly and
deterministically without real wall-clock waits.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

_MONO = "dms.modules.calibrator.time.monotonic"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make() -> object:
    from dms.modules.calibrator import EARCalibrator
    return EARCalibrator(
        duration_s=2.0,
        drowsy_ratio=0.75,
        fallback_ear=0.20,
        fallback_mar=0.55,
        min_samples=3,
    )


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_initial_done_false(self):
        c = _make()
        assert c.done is False

    def test_initial_ear_threshold_is_fallback(self):
        c = _make()
        assert c.ear_threshold == pytest.approx(0.20)

    def test_initial_mar_threshold_is_fallback(self):
        c = _make()
        assert c.mar_threshold == pytest.approx(0.55)

    def test_progress_zero_before_start(self):
        c = _make()
        with patch(_MONO, return_value=100.0):
            assert c.progress == pytest.approx(0.0)

    def test_custom_params_stored(self):
        from dms.modules.calibrator import EARCalibrator
        c = EARCalibrator(
            duration_s=10.0,
            drowsy_ratio=0.80,
            fallback_ear=0.25,
            fallback_mar=0.60,
            min_samples=10,
        )
        assert c._duration_s   == pytest.approx(10.0)
        assert c._drowsy_ratio == pytest.approx(0.80)
        assert c._fallback_ear == pytest.approx(0.25)
        assert c._fallback_mar == pytest.approx(0.60)
        assert c._min_samples  == 10


# ---------------------------------------------------------------------------
# 2. update() -- early-return guards
# ---------------------------------------------------------------------------

class TestUpdateGuards:
    def test_update_noop_when_done(self):
        c = _make()
        c._done = True
        c.update(ear=0.30, mar=0.50)   # should be a silent no-op
        assert c._ear_samples == []

    def test_update_skips_non_positive_ear(self):
        with patch(_MONO, return_value=0.0):
            c = _make()
        with patch(_MONO, return_value=0.0):
            c.update(ear=0.0,  mar=0.5)
            c.update(ear=-0.1, mar=0.5)
        assert c._ear_samples == []
        assert c._start is None

    def test_update_first_valid_sets_start(self):
        c = _make()
        t0 = 50.0
        with patch(_MONO, return_value=t0):
            c.update(ear=0.30, mar=0.50)
        assert c._start == pytest.approx(t0)

    def test_update_appends_sample(self):
        c = _make()
        with patch(_MONO, return_value=0.0):
            c.update(ear=0.30, mar=0.50)
        assert len(c._ear_samples) == 1
        assert c._ear_samples[0] == pytest.approx(0.30)

    def test_update_does_not_finalise_before_duration(self):
        c = _make()
        with patch(_MONO, side_effect=[0.0, 1.0]):   # start=0, elapsed=1 < 2
            c.update(ear=0.30, mar=0.50)
        assert c.done is False


# ---------------------------------------------------------------------------
# 3. _finalise() -- both branches
# ---------------------------------------------------------------------------

class TestFinalise:
    def test_finalise_with_enough_samples(self):
        """Mean of [0.30, 0.32, 0.28] * 0.75 = ~0.225."""
        c = _make()
        samples = [0.30, 0.32, 0.28]
        expected = round(sum(samples) / len(samples) * 0.75, 4)

        # Feed samples at t=0,0,0 then trigger at t=3 (> duration_s=2)
        times = [0.0, 0.0, 0.0, 3.0]
        with patch(_MONO, side_effect=times):
            c.update(ear=0.30, mar=0.5)
            c.update(ear=0.32, mar=0.5)
            c.update(ear=0.28, mar=0.5)  # triggers _finalise (elapsed >= 2)
        assert c.done is True
        assert c.ear_threshold == pytest.approx(expected, abs=1e-4)

    def test_finalise_fallback_when_too_few_samples(self):
        """Only 1 sample when min_samples=3 -> use fallback."""
        c = _make()   # min_samples=3
        # start=0, second call elapsed=5 (> duration_s=2)
        with patch(_MONO, side_effect=[0.0, 5.0]):
            c.update(ear=0.30, mar=0.5)
        assert c.done is True
        assert c.ear_threshold == pytest.approx(0.20)  # fallback

    def test_finalise_called_once(self):
        """Second update after done should not re-run _finalise."""
        c = _make()
        with patch(_MONO, side_effect=[0.0, 0.0, 0.0, 5.0]):
            c.update(ear=0.30, mar=0.5)
            c.update(ear=0.30, mar=0.5)
            c.update(ear=0.30, mar=0.5)   # finalises here
        thresh_after = c.ear_threshold
        c.update(ear=0.99, mar=0.9)       # done=True -> no-op
        assert c.ear_threshold == pytest.approx(thresh_after)


# ---------------------------------------------------------------------------
# 4. progress property
# ---------------------------------------------------------------------------

class TestProgress:
    def test_progress_one_when_done(self):
        c = _make()
        c._done = True
        assert c.progress == pytest.approx(1.0)

    def test_progress_zero_before_start(self):
        c = _make()
        assert c.progress == pytest.approx(0.0)

    def test_progress_mid_collection(self):
        c = _make()   # duration_s=2.0
        with patch(_MONO, side_effect=[10.0, 11.0, 11.0]):   # start=10, now=11 -> 0.5
            c.update(ear=0.30, mar=0.5)
            p = c.progress
        assert 0.0 < p <= 1.0

    def test_progress_clamped_at_one(self):
        """progress should never exceed 1.0."""
        c = _make()   # duration_s=2.0
        c._start = 0.0
        c._done  = False
        with patch(_MONO, return_value=999.0):
            assert c.progress == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. Properties -- ear/mar threshold after finalise
# ---------------------------------------------------------------------------

class TestProperties:
    def test_ear_threshold_after_calibration(self):
        c = _make()
        c._ear_threshold = 0.22
        assert c.ear_threshold == pytest.approx(0.22)

    def test_mar_threshold_unchanged(self):
        """MAR threshold is only ever the fallback value."""
        c = _make()
        assert c.mar_threshold == pytest.approx(0.55)
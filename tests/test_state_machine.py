"""
tests/test_state_machine.py
============================
Coverage target: = 90% of src/dms/modules/state_machine.py

AlertController is fully mocked so no GPIO/threads spin up.
time.monotonic() is patched for deterministic timing.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

_MONO    = "dms.modules.state_machine.time.monotonic"
_THREAD  = "dms.modules.alert.threading.Thread"
_SLEEP   = "dms.modules.alert.time.sleep"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_alert_mock():
    """Return a mock AlertController that records all calls."""
    alert = MagicMock()
    alert.set_normal   = MagicMock()
    alert.set_warning  = MagicMock()
    alert.set_critical = MagicMock()
    return alert


def _make_sm(warning_dur=2.0, critical_dur=3.0, log_interval=1.0):
    alert = _make_alert_mock()
    with patch(_MONO, return_value=0.0):
        from dms.modules.state_machine import DMSStateMachine
        sm = DMSStateMachine(
            alert=alert,
            warning_duration=warning_dur,
            critical_duration=critical_dur,
            log_interval=log_interval,
        )
    return sm, alert


def _event(drowsy=False, yawning=False, distracted=False,
           phone=False, imu_tilt=False,
           ear=0.3, mar=0.3, yaw=0.0, pitch=0.0, roll=0.0):
    from dms.modules.state_machine import DMSEvent
    return DMSEvent(
        drowsy=drowsy, yawning=yawning, distracted=distracted,
        phone=phone, imu_tilt=imu_tilt,
        ear=ear, mar=mar, yaw=yaw, pitch=pitch, roll=roll,
    )


# ---------------------------------------------------------------------------
# 1. DMSEvent dataclass
# ---------------------------------------------------------------------------

class TestDMSEvent:
    def test_default_any_event_false(self):
        e = _event()
        assert e.any_event is False

    def test_drowsy_sets_any_event(self):
        e = _event(drowsy=True)
        assert e.any_event is True

    def test_all_flags_any_event(self):
        from dms.modules.state_machine import DMSEvent
        e = DMSEvent(drowsy=True, yawning=True, distracted=True,
                     phone=True, imu_tilt=True)
        assert e.any_event is True

    def test_active_labels_empty_when_no_event(self):
        e = _event()
        assert e.active_labels == []

    def test_active_labels_correct(self):
        e = _event(drowsy=True, phone=True)
        labels = e.active_labels
        assert "DROWSY" in labels
        assert "PHONE"  in labels
        assert len(labels) == 2

    def test_all_active_labels(self):
        e = _event(drowsy=True, yawning=True, distracted=True,
                   phone=True, imu_tilt=True)
        assert set(e.active_labels) == {
            "DROWSY", "YAWNING", "DISTRACTED", "PHONE", "IMU_TILT"
        }


# ---------------------------------------------------------------------------
# 2. DMSStateMachine construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_initial_state_normal(self):
        from dms.modules.state_machine import DMSState
        sm, alert = _make_sm()
        assert sm.state == DMSState.NORMAL

    def test_alert_set_normal_called_on_init(self):
        sm, alert = _make_sm()
        alert.set_normal.assert_called_once()

    def test_state_name_property(self):
        sm, _ = _make_sm()
        assert sm.state_name == "NORMAL"


# ---------------------------------------------------------------------------
# 3. update() -- NORMAL state transitions
# ---------------------------------------------------------------------------

class TestUpdateFromNormal:
    def test_stays_normal_when_no_event(self):
        from dms.modules.state_machine import DMSState
        sm, alert = _make_sm()
        with patch(_MONO, return_value=0.1):
            sm.update(_event())
        assert sm.state == DMSState.NORMAL

    def test_transitions_to_warning_on_event(self):
        from dms.modules.state_machine import DMSState
        sm, alert = _make_sm()
        with patch(_MONO, return_value=0.1):
            sm.update(_event(drowsy=True))
        assert sm.state == DMSState.WARNING
        alert.set_warning.assert_called_once()


# ---------------------------------------------------------------------------
# 4. update() -- WARNING state transitions
# ---------------------------------------------------------------------------

class TestUpdateFromWarning:
    def _in_warning(self):
        sm, alert = _make_sm(warning_dur=2.0)
        with patch(_MONO, return_value=0.1):
            sm.update(_event(drowsy=True))   # -> WARNING
        return sm, alert

    def test_warning_back_to_normal_when_event_clears(self):
        from dms.modules.state_machine import DMSState
        sm, alert = self._in_warning()
        with patch(_MONO, return_value=0.5):   # elapsed=0.4, no event
            sm.update(_event())
        assert sm.state == DMSState.NORMAL
        alert.set_normal.assert_called()

    def test_warning_stays_warning_before_duration(self):
        from dms.modules.state_machine import DMSState
        sm, alert = self._in_warning()
        # elapsed = 1.0 < warning_dur=2.0
        with patch(_MONO, return_value=1.1):
            sm.update(_event(drowsy=True))
        assert sm.state == DMSState.WARNING

    def test_warning_escalates_to_critical_after_duration(self):
        from dms.modules.state_machine import DMSState
        sm, alert = self._in_warning()
        # elapsed = 2.5 >= warning_dur=2.0
        with patch(_MONO, return_value=2.6):
            sm.update(_event(drowsy=True))
        assert sm.state == DMSState.CRITICAL
        alert.set_critical.assert_called_once()


# ---------------------------------------------------------------------------
# 5. update() -- CRITICAL state transitions
# ---------------------------------------------------------------------------

class TestUpdateFromCritical:
    def _in_critical(self):
        sm, alert = _make_sm(warning_dur=0.0, critical_dur=3.0)
        # Jump straight to CRITICAL via two update calls
        with patch(_MONO, return_value=0.0):
            sm.update(_event(drowsy=True))   # NORMAL -> WARNING (dur=0)
        with patch(_MONO, return_value=0.1):
            sm.update(_event(drowsy=True))   # WARNING -> CRITICAL
        return sm, alert

    def test_critical_back_to_normal_when_event_clears(self):
        from dms.modules.state_machine import DMSState
        sm, alert = self._in_critical()
        with patch(_MONO, return_value=0.5):
            sm.update(_event())
        assert sm.state == DMSState.NORMAL

    def test_critical_stays_critical_before_duration(self):
        from dms.modules.state_machine import DMSState
        sm, alert = self._in_critical()
        with patch(_MONO, return_value=0.5):   # elapsed < critical_dur=3.0
            sm.update(_event(drowsy=True))
        assert sm.state == DMSState.CRITICAL

    def test_critical_re_triggers_after_duration(self):
        """Re-trigger: state stays CRITICAL but _state_entered_at resets."""
        from dms.modules.state_machine import DMSState
        sm, alert = self._in_critical()
        entered_before = sm._state_entered_at
        with patch(_MONO, return_value=entered_before + 4.0):  # > critical_dur=3
            sm.update(_event(drowsy=True))
        assert sm.state == DMSState.CRITICAL
        assert sm._state_entered_at == pytest.approx(entered_before + 4.0)


# ---------------------------------------------------------------------------
# 6. update() -- logging branch
# ---------------------------------------------------------------------------

class TestUpdateLogging:
    def test_logging_fires_when_interval_elapsed(self, caplog):
        import logging
        sm, _ = _make_sm(log_interval=0.0)   # always log
        with caplog.at_level(logging.INFO):
            with patch(_MONO, return_value=0.0):
                sm.update(_event(drowsy=True))
            with patch(_MONO, return_value=1.0):
                sm.update(_event(drowsy=True))
        assert any("State" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# 7. _transition() -- same-state no-op
# ---------------------------------------------------------------------------

class TestTransitionNoOp:
    def test_no_double_call_for_same_state(self):
        from dms.modules.state_machine import DMSState
        sm, alert = _make_sm()
        alert.set_normal.reset_mock()
        with patch(_MONO, return_value=0.5):
            sm._transition(DMSState.NORMAL, 0.5)   # already NORMAL
        alert.set_normal.assert_not_called()


# ---------------------------------------------------------------------------
# 8. Properties
# ---------------------------------------------------------------------------

class TestProperties:
    def test_time_in_state_positive(self):
        sm, _ = _make_sm()
        with patch(_MONO, return_value=5.0):
            t = sm.time_in_state
        assert t >= 0.0

    def test_state_name_warning(self):
        sm, _ = _make_sm()
        with patch(_MONO, return_value=0.0):
            sm.update(_event(drowsy=True))
        assert sm.state_name == "WARNING"


# ---------------------------------------------------------------------------
# 9. stats() / log_stats() / reset()
# ---------------------------------------------------------------------------

class TestStats:
    def test_stats_keys_present(self):
        sm, _ = _make_sm()
        s = sm.stats()
        assert "session_duration_s" in s
        assert "warning_count"      in s
        assert "critical_count"     in s
        assert "current_state"      in s

    def test_warning_count_increments(self):
        sm, _ = _make_sm()
        with patch(_MONO, return_value=0.0):
            sm.update(_event(drowsy=True))    # -> WARNING
        assert sm.stats()["warning_count"] == 1

    def test_critical_count_increments(self):
        sm, _ = _make_sm(warning_dur=0.0)
        with patch(_MONO, return_value=0.0):
            sm.update(_event(drowsy=True))    # -> WARNING (dur=0)
        with patch(_MONO, return_value=0.1):
            sm.update(_event(drowsy=True))    # -> CRITICAL
        assert sm.stats()["critical_count"] == 1

    def test_log_stats_does_not_raise(self, caplog):
        import logging
        sm, _ = _make_sm()
        with caplog.at_level(logging.INFO):
            sm.log_stats()

    def test_reset_goes_to_normal(self):
        from dms.modules.state_machine import DMSState
        sm, alert = _make_sm(warning_dur=0.0)
        with patch(_MONO, return_value=0.0):
            sm.update(_event(drowsy=True))   # -> WARNING
        with patch(_MONO, return_value=0.5):
            sm.reset()
        assert sm.state == DMSState.NORMAL
        alert.set_normal.assert_called()
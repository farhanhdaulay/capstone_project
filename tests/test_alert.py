"""
tests/test_alert.py
===================
Pytest suite for src/dms/modules/alert.py
Targets = 90 % line coverage without hanging.

Why tests hung before
---------------------
set_warning() / set_critical() spin up real background threads that call
time.sleep() in a loop.  If the test never called cleanup() those threads
kept running and pytest blocked waiting for them.

Fix strategy
------------
1. Always run AlertController in mock=True mode (no GPIO).
2. Patch `threading.Thread` so threads are recorded but *not* started,
   eliminating every time.sleep() inside _vib_pulse / _vib_continuous /
   _play_sound_loop from the test process.
3. Patch `time.sleep` as a safety-net for any remaining sleeps.
4. Use the context-manager form (with AlertController(...) as ac:) so
   cleanup() is guaranteed even if a test raises.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, call, patch

import pytest

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _make_ac(**kwargs):
    """Return a mock=True AlertController with threads disabled."""
    from dms.modules.alert import AlertController
    kwargs.setdefault("mock", True)   # ? only sets mock=True if not already provided
    return AlertController(**kwargs)


# Patch target for Thread: we want to intercept every `threading.Thread(...)`
# call inside the alert module so real threads never start.
_THREAD_PATH  = "dms.modules.alert.threading.Thread"
_SLEEP_PATH   = "dms.modules.alert.time.sleep"
_OS_SYS_PATH  = "dms.modules.alert.os.system"


# ---------------------------------------------------------------------------
# 1. AlertState enum
# ---------------------------------------------------------------------------

class TestAlertState:
    def test_enum_values_exist(self):
        from dms.modules.alert import AlertState
        assert AlertState.NORMAL
        assert AlertState.WARNING
        assert AlertState.CRITICAL

    def test_enum_distinct(self):
        from dms.modules.alert import AlertState
        states = {AlertState.NORMAL, AlertState.WARNING, AlertState.CRITICAL}
        assert len(states) == 3


# ---------------------------------------------------------------------------
# 2. Construction
# ---------------------------------------------------------------------------

class TestConstruction:
    def test_default_state_is_normal(self):
        with _make_ac() as ac:
            from dms.modules.alert import AlertState
            assert ac.state == AlertState.NORMAL

    def test_mock_flag_forced_when_no_gpio(self):
        """mock=False should still result in _mock=True when GPIO absent."""
        with patch("dms.modules.alert._GPIO_OK", False):
            with _make_ac(mock=False) as ac:
                assert ac._mock is True

    def test_custom_pins_stored(self):
        with _make_ac(pin_green=11, pin_yellow=13,
                      pin_red=15, pin_vib=19) as ac:
            assert ac._pin_green  == 11
            assert ac._pin_yellow == 13
            assert ac._pin_red    == 15
            assert ac._pin_vib    == 19

    def test_sound_file_stored(self):
        with _make_ac(sound_file="/tmp/beep.wav") as ac:
            assert ac._sound_file == "/tmp/beep.wav"

    def test_default_pins(self):
        with _make_ac() as ac:
            assert ac._pin_green  == 7
            assert ac._pin_yellow == 31
            assert ac._pin_red    == 29
            assert ac._pin_vib    == 33


# ---------------------------------------------------------------------------
# 3. _set_pin  (mock branch  just logs, no GPIO)
# ---------------------------------------------------------------------------

class TestSetPin:
    def test_set_pin_high_mock(self, caplog):
        import logging
        with _make_ac() as ac:
            with caplog.at_level(logging.DEBUG):
                ac._set_pin(15, True)
        assert "HIGH" in caplog.text

    def test_set_pin_low_mock(self, caplog):
        import logging
        with _make_ac() as ac:
            with caplog.at_level(logging.DEBUG):
                ac._set_pin(15, False)
        assert "LOW" in caplog.text

    def test_set_pin_real_calls_gpio(self):
        mock_gpio = MagicMock()
        with patch("dms.modules.alert._GPIO_OK", True), \
             patch("dms.modules.alert.GPIO", mock_gpio, create=True):
            from dms.modules.alert import AlertController
            ac = AlertController(mock=False)
            ac._mock = False
            ac._set_pin(15, True)
            mock_gpio.output.assert_called_with(15, mock_gpio.HIGH)  # ? not assert_called_once_with

# ---------------------------------------------------------------------------
# 4. set_normal
# ---------------------------------------------------------------------------

class TestSetNormal:
    def test_state_becomes_normal(self):
        from dms.modules.alert import AlertState
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            t = MagicMock()
            MockThread.return_value = t
            with _make_ac() as ac:
                ac.set_warning()   # move away from NORMAL first
                ac.set_normal()
                assert ac.state == AlertState.NORMAL

    def test_idempotent_when_already_normal(self):
        from dms.modules.alert import AlertState
        with _make_ac() as ac:
            ac.set_normal()        # already NORMAL
            assert ac.state == AlertState.NORMAL

    def test_green_led_set_after_normal(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.set_warning()
                with patch.object(ac, "_set_pin") as spy:
                    ac.set_normal()
                    # last call should be green=True
                    spy.assert_any_call(ac._pin_green, True)


# ---------------------------------------------------------------------------
# 5. set_warning
# ---------------------------------------------------------------------------

class TestSetWarning:
    def test_state_becomes_warning(self):
        from dms.modules.alert import AlertState
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.set_warning()
                assert ac.state == AlertState.WARNING

    def test_idempotent_when_already_warning(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            mock_t = MagicMock()
            MockThread.return_value = mock_t
            with _make_ac() as ac:
                ac.set_warning()
                call_count = MockThread.call_count
                ac.set_warning()                  # second call  no-op
                assert MockThread.call_count == call_count

    def test_yellow_led_on(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                with patch.object(ac, "_set_pin") as spy:
                    ac.set_warning()
                    spy.assert_any_call(ac._pin_yellow, True)

    def test_transitions_from_critical(self):
        """WARNING after CRITICAL: sound and old vib must be stopped."""
        from dms.modules.alert import AlertState
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.set_critical()
                ac.set_warning()
                assert ac.state == AlertState.WARNING

    def test_vib_thread_started(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            mock_t = MagicMock()
            MockThread.return_value = mock_t
            with _make_ac() as ac:
                ac.set_warning()
                mock_t.start.assert_called()


# ---------------------------------------------------------------------------
# 6. set_critical
# ---------------------------------------------------------------------------

class TestSetCritical:
    def test_state_becomes_critical(self):
        from dms.modules.alert import AlertState
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.set_critical()
                assert ac.state == AlertState.CRITICAL

    def test_idempotent_when_already_critical(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            mock_t = MagicMock()
            MockThread.return_value = mock_t
            with _make_ac() as ac:
                ac.set_critical()
                n = MockThread.call_count
                ac.set_critical()
                assert MockThread.call_count == n

    def test_red_led_on(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                with patch.object(ac, "_set_pin") as spy:
                    ac.set_critical()
                    spy.assert_any_call(ac._pin_red, True)

    def test_both_threads_started(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            mock_t = MagicMock()
            MockThread.return_value = mock_t
            with _make_ac() as ac:
                ac.set_critical()
                # vib thread + sound thread = 2 starts
                assert mock_t.start.call_count >= 2


# ---------------------------------------------------------------------------
# 7. set_state dispatcher
# ---------------------------------------------------------------------------

class TestSetState:
    @pytest.mark.parametrize("state_name,setter", [
        ("NORMAL",   "set_normal"),
        ("WARNING",  "set_warning"),
        ("CRITICAL", "set_critical"),
    ])
    def test_delegates_to_correct_setter(self, state_name, setter):
        from dms.modules.alert import AlertState
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                with patch.object(ac, setter) as mock_setter:
                    ac.set_state(getattr(AlertState, state_name))
                    mock_setter.assert_called_once()


# ---------------------------------------------------------------------------
# 8. _stop_vibration / _stop_sound_alarm
# ---------------------------------------------------------------------------

class TestStopHelpers:
    def test_stop_vibration_sets_pin_low(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.set_warning()
                with patch.object(ac, "_set_pin") as spy:
                    ac._stop_vibration()
                    spy.assert_any_call(ac._pin_vib, False)

    def test_stop_sound_clears_event(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.set_critical()
                ac._stop_sound_alarm()
                # After stop the event should be cleared (ready for reuse)
                assert not ac._stop_sound.is_set()


# ---------------------------------------------------------------------------
# 9. Sound / vibration internal loops (unit-tested in isolation)
# ---------------------------------------------------------------------------

class TestInternalLoops:
    def test_vib_pulse_calls_set_pin(self):
        with patch(_SLEEP_PATH):
            with _make_ac() as ac:
                with patch.object(ac, "_set_pin") as spy:
                    ac._vib_pulse()
                    # motor turned on at least once
                    spy.assert_any_call(ac._pin_vib, True)

    def test_vib_pulse_stops_early_when_event_set(self):
        with patch(_SLEEP_PATH):
            with _make_ac() as ac:
                ac._stop_vib.set()          # signal stop before running
                with patch.object(ac, "_set_pin") as spy:
                    ac._vib_pulse()
                    # should bail out immediately  no HIGH call
                    high_calls = [c for c in spy.call_args_list
                                  if c == call(ac._pin_vib, True)]
                    assert len(high_calls) == 0

    def test_vib_continuous_stops_on_event(self):
        """_vib_continuous loop exits when _stop_vib is pre-set."""
        with patch(_SLEEP_PATH):
            with _make_ac() as ac:
                ac._stop_vib.set()
                with patch.object(ac, "_set_pin") as spy:
                    ac._vib_continuous()
                    spy.assert_any_call(ac._pin_vib, False)  # cleanup call

    def test_play_sound_loop_mock_logs(self, caplog):
        """In mock mode the sound loop should log and sleep, not crash."""
        import logging
        stop_ev = threading.Event()
        with patch(_SLEEP_PATH) as mock_sleep:
            # Make sleep set the stop event so the loop runs exactly once
            def _one_shot(*_):
                stop_ev.set()
            mock_sleep.side_effect = _one_shot

            with _make_ac() as ac:
                ac._stop_sound = stop_ev   # replace with our event
                with caplog.at_level(logging.INFO):
                    ac._play_sound_loop()
        assert "CRITICAL alarm" in caplog.text

    def test_play_sound_loop_uses_aplay_fallback(self, tmp_path):
        wav = tmp_path / "alarm.wav"
        wav.write_bytes(b"RIFF")
    
        stop_ev = threading.Event()
        call_count = 0
    
        # Accept *args and **kwargs because subprocess.run signature is different
        def _subprocess_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            stop_ev.set()  # Kills the loop
    
        mock_gpio = MagicMock()                                        
        with patch("dms.modules.alert._SOUND_OK", False), \
             patch("dms.modules.alert._GPIO_OK", True),\
             patch("dms.modules.alert.GPIO", mock_gpio, create=True),\
             patch("dms.modules.alert.subprocess.run", side_effect=_subprocess_side_effect), \
             patch(_SLEEP_PATH):
            with _make_ac(sound_file=str(wav)) as ac:
                ac._mock        = False
                ac._stop_sound  = stop_ev
                ac._play_sound_loop()
    
        assert call_count >= 1


# ---------------------------------------------------------------------------
# 10. context manager & cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_context_manager_calls_cleanup(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            ac = _make_ac()
            with patch.object(ac, "cleanup") as mock_cleanup:
                with ac:
                    pass
                mock_cleanup.assert_called_once()

    def test_cleanup_turns_off_all_leds_and_vib(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.set_critical()
                with patch.object(ac, "_set_pin") as spy:
                    ac.cleanup()
                    off_calls = [c for c in spy.call_args_list
                                 if c.args[1] is False]
                    assert len(off_calls) >= 4   # 3 LEDs + vib

    def test_cleanup_calls_gpio_cleanup_when_real(self):
        mock_gpio = MagicMock()
        with patch("dms.modules.alert._GPIO_OK", True), \
             patch("dms.modules.alert.GPIO", mock_gpio, create=True), \
             patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            from dms.modules.alert import AlertController
            ac = AlertController(mock=False)
            ac._mock = False
            ac.cleanup()
            mock_gpio.cleanup.assert_called_once()

    def test_double_cleanup_does_not_raise(self):
        with patch(_THREAD_PATH) as MockThread, \
             patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.cleanup()   # first
                ac.cleanup()   # second  must not raise


# ---------------------------------------------------------------------------
# 11. state property
# ---------------------------------------------------------------------------

class TestStateProperty:
    def test_state_property_reflects_set_normal(self):
        from dms.modules.alert import AlertState
        with patch(_THREAD_PATH) as MockThread, patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.set_warning()
                ac.set_normal()
                assert ac.state == AlertState.NORMAL

    def test_state_property_reflects_set_warning(self):
        from dms.modules.alert import AlertState
        with patch(_THREAD_PATH) as MockThread, patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.set_warning()
                assert ac.state == AlertState.WARNING

    def test_state_property_reflects_set_critical(self):
        from dms.modules.alert import AlertState
        with patch(_THREAD_PATH) as MockThread, patch(_SLEEP_PATH):
            MockThread.return_value = MagicMock()
            with _make_ac() as ac:
                ac.set_critical()
                assert ac.state == AlertState.CRITICAL
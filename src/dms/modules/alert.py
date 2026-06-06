# src/dms/modules/alert.py
"""
alert.py -- 3-state alert actuator controller.

Hardware on Jetson Orin Nano Super (BOARD pin numbering):
  Pin 15 -> Green  LED  (NORMAL   state)
  Pin 29 -> Yellow LED  (WARNING  state)
  Pin 31 -> Red    LED  (CRITICAL state)
  Pin 33 -> Vibration motor via 2N7000 MOSFET (WARNING + CRITICAL)
  USB Sound card -> plays alarm via sounddevice / aplay (CRITICAL only)

States
------
  NORMAL   : green  LED on, others off, motor off, no sound
  WARNING  : yellow LED on, short vibration pulse,  no sound
  CRITICAL : red    LED on, continuous vibration,   looping alarm sound

ALERT_MOCK = True  -> prints to console, skips real GPIO (safe for testing).
ALERT_MOCK = False -> drives real hardware.
"""

from __future__ import annotations

import math
import logging
import os
import threading
import time
import subprocess
from enum import Enum, auto
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import Jetson.GPIO as GPIO   # type: ignore
    _GPIO_OK = True
except (ImportError, RuntimeError):
    GPIO = None
    _GPIO_OK = False
    logger.warning("Jetson.GPIO not available -- alert will run in MOCK mode.")

try:
    import sounddevice as sd     # type: ignore
    import numpy as np
    _SOUND_OK = True
except ImportError:
    _SOUND_OK = False
    logger.warning("sounddevice not available -- audio alert will use aplay fallback.")

class AlertState(Enum):
    NORMAL   = auto()
    WARNING  = auto()
    CRITICAL = auto()

class AlertController:
    """
    Controls green/yellow/red LEDs, vibration motor, and USB buzzer.

    Parameters
    ----------
    pin_green  : Physical (BOARD) pin number for green  LED.
    pin_yellow : Physical (BOARD) pin number for yellow LED.
    pin_red    : Physical (BOARD) pin number for red    LED.
    pin_vib    : Physical (BOARD) pin number for vibration motor gate.
    mock       : Skip real GPIO when True (or when Jetson.GPIO absent).
    sound_file : Optional WAV file path for CRITICAL alarm.
                 If None, a synthetic beep tone is generated.
    """

    def __init__(
        self,
        pin_green:  int = 7,
        pin_yellow: int = 31,
        pin_red:    int = 29,
        pin_vib:    int = 33,
        mock:       bool = True,
        sound_file: Optional[str] = None,
    ) -> None:
        self._mock       = mock or not _GPIO_OK
        self._pin_green  = pin_green
        self._pin_yellow = pin_yellow
        self._pin_red    = pin_red
        self._pin_vib    = pin_vib
        self._sound_file = sound_file
        self._state      = AlertState.NORMAL

        self._vib_thread:   Optional[threading.Thread] = None
        self._sound_thread: Optional[threading.Thread] = None
        self._stop_vib      = threading.Event()
        self._stop_sound    = threading.Event()

        self._setup_gpio()
        logger.info(
            "AlertController ready -- mock=%s pins(G=%d Y=%d R=%d V=%d)",
            self._mock, pin_green, pin_yellow, pin_red, pin_vib,
        )

    def _setup_gpio(self) -> None:
        if self._mock:
            logger.info("AlertController: MOCK mode -- no real GPIO used.")
            return
        # BOARD mode = physical pin numbers, correct for Jetson Orin Nano Super
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        for pin in (self._pin_green, self._pin_yellow,
                    self._pin_red,   self._pin_vib):
            GPIO.setup(pin, GPIO.OUT, initial=GPIO.LOW)
        # Safe startup: green LED on
        GPIO.output(self._pin_green, GPIO.HIGH)
        logger.info("GPIO pins configured in BOARD mode.")

    def _set_pin(self, pin: int, high: bool) -> None:
        if self._mock:
            logger.debug("  [MOCK GPIO] pin %d -> %s", pin, "HIGH" if high else "LOW")
            return
        GPIO.output(pin, GPIO.HIGH if high else GPIO.LOW)

    def _all_leds_off(self) -> None:
        for pin in (self._pin_green, self._pin_yellow, self._pin_red):
            self._set_pin(pin, False)

    def _vib_pulse(self) -> None:
        """Double-pulse for WARNING -- runs in background thread."""
        for _ in range(2):
            if self._stop_vib.is_set():
                break
            self._set_pin(self._pin_vib, True)
            time.sleep(0.15)
            self._set_pin(self._pin_vib, False)
            time.sleep(0.10)

    def _vib_continuous(self) -> None:
        """Repeating on/off for CRITICAL -- runs until stop event."""
        while not self._stop_vib.is_set():
            self._set_pin(self._pin_vib, True)
            time.sleep(0.3)
            if self._stop_vib.is_set():
                break
            self._set_pin(self._pin_vib, False)
            time.sleep(0.1)
        self._set_pin(self._pin_vib, False)

    def _stop_vibration(self) -> None:
        self._stop_vib.set()
        if self._vib_thread and self._vib_thread.is_alive():
            self._vib_thread.join(timeout=1.0)
        self._set_pin(self._pin_vib, False)
        self._stop_vib.clear()

    def _beep_siren(self, sample_rate: int = 44100) -> None:
        """Two-tone siren using sounddevice -- no files needed."""
        duration = 0.35
        for freq in (1200.0, 800.0):
            if self._stop_sound.is_set():
                return
            n    = int(sample_rate * duration)
            wave = np.array([
                min(i, n - i, 500) / 500.0
                * 0.7 * math.sin(2 * math.pi * freq * i / sample_rate)
                for i in range(n)
            ], dtype=np.float32)
            sd.play(wave, samplerate=sample_rate, blocking=True)
    
    def _make_alert_wav(self, path: str = "/tmp/dms_alert.wav") -> str:  # nosec B108
        """Generate a two-tone siren WAV file and return its path."""
        import wave
        import struct
        sample_rate = 44100
        duration    = 0.4
        tones       = [1200.0, 800.0]
        repeats     = 3
    
        frames = []
        for _ in range(repeats):
            for freq in tones:
                n = int(sample_rate * duration)
                for i in range(n):
                    env = min(i, n - i, 500) / 500.0
                    val = env * 0.8 * math.sin(2 * math.pi * freq * i / sample_rate)
                    frames.append(struct.pack("<h", int(val * 32767)))
    
        with wave.open(path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        return path

    def _play_sound_loop(self) -> None:
        """Loop alarm sound until stop event -- runs in background thread."""
        alert_wav = self._sound_file
        if not alert_wav or not os.path.isfile(alert_wav):
            try:
                alert_wav = self._make_alert_wav()
            except Exception as exc:
                logger.error("Could not generate alert WAV: %s", exc)
                alert_wav = None
    
        while not self._stop_sound.is_set():
            if self._mock:
                logger.info("[ALERT SOUND] CRITICAL alarm")
                time.sleep(0.8)
                continue
    
            if _SOUND_OK:
                try:
                    self._beep_siren()
                    continue
                except Exception as exc:
                    logger.error("sounddevice error: %s", exc)
    
            if alert_wav and os.path.isfile(alert_wav):
                subprocess.run(
                    ["aplay", "-q", alert_wav],
                    stderr=subprocess.DEVNULL, check=False
                )
            else:
                time.sleep(0.8)

    def _stop_sound_alarm(self) -> None:
        self._stop_sound.set()
        if self._sound_thread and self._sound_thread.is_alive():
            self._sound_thread.join(timeout=2.0)
        if _SOUND_OK and not self._mock:
            try:
                sd.stop()
            except Exception:
                pass
        self._stop_sound.clear()

    def set_normal(self) -> None:
        """NORMAL: green LED on, all actuators off."""
        if self._state == AlertState.NORMAL:
            return
        self._state = AlertState.NORMAL
        self._stop_vibration()
        self._stop_sound_alarm()
        self._all_leds_off()
        self._set_pin(self._pin_green, True)
        logger.info("[ALERT] NORMAL -- green LED ON")

    def set_warning(self) -> None:
        """WARNING: yellow LED on, short vibration pulse, no sound."""
        if self._state == AlertState.WARNING:
            return
        prev = self._state
        self._state = AlertState.WARNING
        if prev == AlertState.CRITICAL:
            self._stop_sound_alarm()
            self._stop_vibration()
        self._all_leds_off()
        self._set_pin(self._pin_yellow, True)
        self._stop_vib.clear()
        self._vib_thread = threading.Thread(target=self._vib_pulse, daemon=True)
        self._vib_thread.start()
        logger.info("[ALERT] WARNING -- yellow LED ON + vibration pulse")

    def set_critical(self) -> None:
        if self._state == AlertState.CRITICAL:
            return
        self._state = AlertState.CRITICAL
        self._all_leds_off()           # turn off yellow IMMEDIATELY
        self._set_pin(self._pin_red, True)   # red on right away
        self._stop_vibration()         # now join the old vib thread (yellow already off)
        self._stop_vib.clear()
        self._vib_thread = threading.Thread(target=self._vib_continuous, daemon=True)
        self._vib_thread.start()
        self._stop_sound.clear()
        self._sound_thread = threading.Thread(target=self._play_sound_loop, daemon=True)
        self._sound_thread.start()
        logger.info("[ALERT] CRITICAL -- red LED ON + vibration + buzzer")
    def set_state(self, state: AlertState) -> None:
        """Set state from enum value."""
        if state == AlertState.NORMAL:
            self.set_normal()
        elif state == AlertState.WARNING:
            self.set_warning()
        elif state == AlertState.CRITICAL:
            self.set_critical()

    @property
    def state(self) -> AlertState:
        return self._state

    def cleanup(self) -> None:
        """Stop all actuators and release GPIO. Call on exit."""
        self._stop_vibration()
        self._stop_sound_alarm()
        self._all_leds_off()
        self._set_pin(self._pin_vib, False)
        if not self._mock and _GPIO_OK:
            try:
                GPIO.cleanup()
            except Exception:
                pass
        logger.info("AlertController cleaned up.")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()
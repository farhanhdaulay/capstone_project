"""
state_machine.py
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from enum import Enum, auto

from dms.modules.alert import AlertController

logger = logging.getLogger(__name__)

@dataclass
class DMSEvent:
    """Snapshot of detector outputs for one frame."""
    drowsy:      bool = False   # EAR below threshold for N frames
    yawning:     bool = False   # MAR above threshold for N frames
    distracted:  bool = False   # Head yaw/pitch beyond threshold
    phone:       bool = False   # Phone detected
    imu_tilt:    bool = False   # IMU roll/pitch beyond threshold (optional)

    ear:    float = 0.0
    mar:    float = 0.0
    yaw:    float = 0.0
    pitch:  float = 0.0
    roll:   float = 0.0

    @property
    def any_event(self) -> bool:
        return self.drowsy or self.yawning or self.distracted or self.phone or self.imu_tilt

    @property
    def active_labels(self) -> list[str]:
        labels = []
        if self.drowsy:
            labels.append("DROWSY")
        if self.yawning:
            labels.append("YAWNING")
        if self.distracted:
            labels.append("DISTRACTED")
        if self.phone:      
            labels.append("PHONE")
        if self.imu_tilt:   
            labels.append("IMU_TILT")
        return labels

class DMSState(Enum):
    NORMAL   = auto()
    WARNING  = auto()
    CRITICAL = auto()

class DMSStateMachine:
    """
    Finite-state machine that maps detector events -> alert states.

    Parameters
    ----------
    alert           : AlertController instance (controls LEDs/motor/sound).
    warning_duration: Seconds in WARNING before escalating to CRITICAL.
    critical_duration: Seconds in CRITICAL before auto-reset to NORMAL.
    log_interval    : Seconds between state-change log entries.
    """

    def __init__(
        self,
        alert: AlertController,
        warning_duration:  float = 2.0,
        critical_duration: float = 3.0,
        log_interval:      float = 1.0,
    ) -> None:
        self._alert             = alert
        self._warning_duration  = warning_duration
        self._critical_duration = critical_duration
        self._log_interval      = log_interval

        self._state             = DMSState.NORMAL
        self._state_entered_at  = time.monotonic()
        self._last_log_at       = 0.0

        # Statistics
        self._warning_count  = 0
        self._critical_count = 0
        self._session_start  = time.monotonic()

        # Apply initial alert state
        self._alert.set_normal()
        logger.info("DMSStateMachine initialised -- state: NORMAL")

    def update(self, event: DMSEvent) -> DMSState:
        """
        Process one frame's events and transition state if needed.

        Call this once per frame after all detectors have run.

        Parameters
        ----------
        event : DMSEvent with flags from all detectors.

        Returns
        -------
        Current DMSState after processing.
        """
        now = time.monotonic()
        elapsed = now - self._state_entered_at

        if self._state == DMSState.NORMAL:
            if event.any_event:
                self._transition(DMSState.WARNING, now)

        elif self._state == DMSState.WARNING:
            if not event.any_event:
                self._transition(DMSState.NORMAL, now)
            elif elapsed >= self._warning_duration:
                self._transition(DMSState.CRITICAL, now)

        elif self._state == DMSState.CRITICAL:
            if not event.any_event:
                self._transition(DMSState.NORMAL, now)
            elif elapsed >= self._critical_duration:
                # Re-trigger alert (keep cycling to re-alert)
                logger.warning(
                    "CRITICAL sustained %.1f s -- re-triggering alert: %s",
                    elapsed, event.active_labels,
                )
                self._state_entered_at = now   # reset timer, stay CRITICAL

        if now - self._last_log_at >= self._log_interval and event.any_event:
            logger.info(
                "State: %-8s | Events: %-30s | EAR=%.3f MAR=%.3f "
                "Yaw=%.1f deg Pitch=%.1f deg",
                self._state.name,
                str(event.active_labels),
                event.ear, event.mar, event.yaw, event.pitch,
            )
            self._last_log_at = now

        return self._state

    def _transition(self, new_state: DMSState, now: float) -> None:
        if new_state == self._state:
            return

        logger.info(
            "State transition: %s -> %s",
            self._state.name, new_state.name,
        )
        self._state            = new_state
        self._state_entered_at = now

        if new_state == DMSState.NORMAL:
            self._alert.set_normal()
        elif new_state == DMSState.WARNING:
            self._warning_count += 1
            self._alert.set_warning()
        elif new_state == DMSState.CRITICAL:
            self._critical_count += 1
            self._alert.set_critical()

    @property
    def state(self) -> DMSState:
        return self._state

    @property
    def state_name(self) -> str:
        return self._state.name

    @property
    def time_in_state(self) -> float:
        """Seconds elapsed in current state."""
        return time.monotonic() - self._state_entered_at

    def stats(self) -> dict:
        return {
            "session_duration_s": time.monotonic() - self._session_start,
            "warning_count":      self._warning_count,
            "critical_count":     self._critical_count,
            "current_state":      self._state.name,
        }

    def log_stats(self) -> None:
        s = self.stats()
        logger.info(
            "Session stats -- duration: %.0f s | warnings: %d | criticals: %d",
            s["session_duration_s"], s["warning_count"], s["critical_count"],
        )

    def reset(self) -> None:
        """Force reset to NORMAL (e.g. after key-off)."""
        self._transition(DMSState.NORMAL, time.monotonic())
        logger.info("DMSStateMachine reset to NORMAL.")
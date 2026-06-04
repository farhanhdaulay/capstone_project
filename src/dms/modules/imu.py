"""
imu.py - MPU6050 I2C reader with dual-axis Kalman filter.

Hardware : MPU6050 on I2C bus 1, address 0x68 (Jetson Orin Nano Super).
Provides : IMUReader - call .read() ? (roll_deg, pitch_deg).

Kalman filter fuses accelerometer angles (noisy but drift-free) with
gyroscope rate (smooth but drifts) for stable roll/pitch estimates.

Usage
-----
    from dms.modules.imu import IMUReader
    imu = IMUReader()
    roll, pitch = imu.read()
    imu.close()
"""

from __future__ import annotations

import math
import time
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# -- Try importing smbus2 (may not be present in simulation env) --------------
try:
    import smbus2 as smbus
    _HW_AVAILABLE = True
except ImportError:
    _HW_AVAILABLE = False
    logger.warning("smbus2 not installed - IMU running in SIMULATION mode.")

# -- MPU-6050 register map ----------------------------------------------------
_REG_PWR_MGMT_1   = 0x6B
_REG_SMPLRT_DIV   = 0x19
_REG_CONFIG       = 0x1A
_REG_GYRO_CONFIG  = 0x1B   # ?250 ?/s  ? scale = 131.0
_REG_ACCEL_CONFIG = 0x1C   # ?2 g      ? scale = 16384.0
_REG_ACCEL_XOUT_H = 0x3B   # 6 bytes: AX, AY, AZ
_REG_GYRO_XOUT_H  = 0x43   # 6 bytes: GX, GY, GZ

_ACCEL_SCALE = 16384.0     # LSB / g
_GYRO_SCALE  = 131.0       # LSB / (?/s)


# -- Kalman filter (one per axis) ---------------------------------------------
@dataclass
class _Kalman:
    """
    Simplified 1-D Kalman filter for angle estimation.
    State  : angle (degrees)
    Input  : gyroscope rate (?/s)
    Measure: accelerometer-derived angle (?)
    """
    Q_angle:   float = 0.001   # process noise - angle
    Q_bias:    float = 0.003   # process noise - gyro bias
    R_measure: float = 0.03    # measurement noise

    angle: float = 0.0
    bias:  float = 0.0

    # 2?2 error covariance matrix (flattened)
    P00: float = 0.0
    P01: float = 0.0
    P10: float = 0.0
    P11: float = 0.0

    def update(self, new_angle: float, new_rate: float, dt: float) -> float:
        """
        Predict + update step.

        Parameters
        ----------
        new_angle : accelerometer-derived angle (degrees)
        new_rate  : gyroscope rate (?/s) on this axis
        dt        : elapsed time since last call (seconds)

        Returns
        -------
        Filtered angle (degrees)
        """
        # -- Predict ----------------------------------------------------------
        rate = new_rate - self.bias
        self.angle += dt * rate

        self.P00 += dt * (dt * self.P11 - self.P01 - self.P10 + self.Q_angle)
        self.P01 -= dt * self.P11
        self.P10 -= dt * self.P11
        self.P11 += self.Q_bias * dt

        # -- Update -----------------------------------------------------------
        S = self.P00 + self.R_measure          # innovation covariance
        K0 = self.P00 / S                      # Kalman gain row 0
        K1 = self.P10 / S                      # Kalman gain row 1

        y = new_angle - self.angle             # innovation
        self.angle += K0 * y
        self.bias  += K1 * y

        P00_tmp = self.P00
        P01_tmp = self.P01
        self.P00 -= K0 * P00_tmp
        self.P01 -= K0 * P01_tmp
        self.P10 -= K1 * P00_tmp
        self.P11 -= K1 * P01_tmp

        return self.angle


# -- Main class ---------------------------------------------------------------
class IMUReader:
    """
    Reads MPU6050 over I2C and returns Kalman-filtered roll/pitch.

    Parameters
    ----------
    bus_num : I2C bus number (default 1 on Jetson).
    address : I2C address (default 0x68).
    """

    def __init__(self, bus_num: int = 1, address: int = 0x68) -> None:
        self._addr = address
        self._bus: object | None = None
        self._sim_mode = not _HW_AVAILABLE

        self._kalman_roll  = _Kalman()
        self._kalman_pitch = _Kalman()
        self._last_time    = time.monotonic()

        # Simulation counter
        self._sim_t = 0.0

        if not self._sim_mode:
            try:
                self._bus = smbus.SMBus(bus_num)
                self._init_mpu()
                logger.info("MPU6050 initialised on bus %d addr 0x%02X",
                            bus_num, address)
            except Exception as exc:
                logger.error("MPU6050 init failed: %s - falling back to SIMULATION", exc)
                self._sim_mode = True
        else:
            logger.info("IMU SIMULATION mode active.")

    # -- Hardware init ---------------------------------------------------------
    def _init_mpu(self) -> None:
        w = self._bus.write_byte_data
        w(self._addr, _REG_PWR_MGMT_1,   0x00)  # wake up
        w(self._addr, _REG_SMPLRT_DIV,   0x07)  # 1 kHz / (7+1) = 125 Hz
        w(self._addr, _REG_CONFIG,        0x00)  # no DLPF
        w(self._addr, _REG_GYRO_CONFIG,   0x00)  # ?250 ?/s
        w(self._addr, _REG_ACCEL_CONFIG,  0x00)  # ?2 g

    # -- Raw read helpers ------------------------------------------------------
    def _read_word_2c(self, reg: int) -> int:
        """Read a signed 16-bit value from two consecutive registers."""
        high = self._bus.read_byte_data(self._addr, reg)
        low  = self._bus.read_byte_data(self._addr, reg + 1)
        val  = (high << 8) | low
        return val - 65536 if val >= 0x8000 else val

    def _read_raw(self) -> tuple[float, float, float, float, float, float]:
        """Return (ax, ay, az) in g and (gx, gy, gz) in ?/s."""
        ax = self._read_word_2c(_REG_ACCEL_XOUT_H)     / _ACCEL_SCALE
        ay = self._read_word_2c(_REG_ACCEL_XOUT_H + 2) / _ACCEL_SCALE
        az = self._read_word_2c(_REG_ACCEL_XOUT_H + 4) / _ACCEL_SCALE
        gx = self._read_word_2c(_REG_GYRO_XOUT_H)      / _GYRO_SCALE
        gy = self._read_word_2c(_REG_GYRO_XOUT_H + 2)  / _GYRO_SCALE
        gz = self._read_word_2c(_REG_GYRO_XOUT_H + 4)  / _GYRO_SCALE
        return ax, ay, az, gx, gy, gz

    # -- Simulation ------------------------------------------------------------
    def _sim_read(self) -> tuple[float, float]:
        """Return slowly oscillating roll/pitch for offline testing."""
        self._sim_t += 0.033
        roll  = 5.0 * math.sin(self._sim_t * 0.3)
        pitch = 3.0 * math.cos(self._sim_t * 0.2)
        return roll, pitch

    # -- Public API ------------------------------------------------------------
    def read(self) -> tuple[float, float]:
        """
        Return Kalman-filtered (roll_deg, pitch_deg).

        roll  > 0 ? tilted right
        pitch > 0 ? tilted forward (nose down)
        """
        if self._sim_mode:
            return self._sim_read()

        now = time.monotonic()
        dt  = now - self._last_time
        dt  = max(dt, 1e-4)   # guard division-by-zero on first call
        self._last_time = now

        try:
            ax, ay, az, gx, gy, _ = self._read_raw()
        except Exception as exc:
            logger.error("IMU read error: %s", exc)
            return 0.0, 0.0

        # Accelerometer-derived angles (atan2 avoids gimbal lock for small tilts)
        accel_roll  = math.degrees(math.atan2(ay, az))
        accel_pitch = math.degrees(math.atan2(-ax, math.sqrt(ay * ay + az * az)))

        roll  = self._kalman_roll.update(accel_roll,  gx, dt)
        pitch = self._kalman_pitch.update(accel_pitch, gy, dt)

        return roll, pitch

    def close(self) -> None:
        """Release I2C bus."""
        if self._bus is not None:
            try:
                self._bus.close()
            except Exception:
                pass
        logger.info("IMU closed.")

    # -- Context manager -------------------------------------------------------
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
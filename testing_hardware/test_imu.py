"""
test_imu.py — MPU6050 IMU reader test.

Prints Kalman-filtered roll/pitch at ~30 Hz for 10 seconds.
Runs in SIMULATION mode when smbus2 is not available or hardware absent.

Run
---
    pdm run python tests/test_imu.py
    pdm run python tests/test_imu.py --duration 20
"""

import argparse
import logging
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dms.modules.imu import IMUReader
import dms.config as cfg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
logger = logging.getLogger("test_imu")


def test_imu(duration: float = 10.0, bus: int = cfg.IMU_BUS,
             addr: int = cfg.IMU_ADDRESS) -> None:
    logger.info("=== IMU Test (bus=%d, addr=0x%02X, duration=%.0f s) ===",
                bus, addr, duration)

    with IMUReader(bus_num=bus, address=addr) as imu:
        t0 = time.monotonic()
        count = 0

        while time.monotonic() - t0 < duration:
            roll, pitch = imu.read()
            count += 1

            # Print at ~5 Hz to stdout (read at full rate for Kalman)
            if count % 6 == 0:
                bar_r = int(abs(roll)  / 90 * 20)
                bar_p = int(abs(pitch) / 90 * 20)
                logger.info(
                    "Roll: %+7.2f°  %s  |  Pitch: %+7.2f°  %s",
                    roll,  ("█" * bar_r).ljust(20),
                    pitch, ("█" * bar_p).ljust(20),
                )

            # Warn if tilt exceeds driving-safe threshold
            tilt_thresh = getattr(cfg, "IMU_TILT_THRESHOLD", 45.0)
            if abs(roll) > tilt_thresh or abs(pitch) > tilt_thresh:
                logger.warning("⚠️  IMU tilt exceeded threshold! roll=%.1f pitch=%.1f",
                               roll, pitch)

            time.sleep(1 / 30)  # ~30 Hz

    elapsed = time.monotonic() - t0
    rate = count / elapsed
    logger.info("Read %.0f samples in %.1f s → %.1f Hz", count, elapsed, rate)
    logger.info("=== IMU Test DONE ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMU sensor test")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="Test duration in seconds (default 10).")
    parser.add_argument("--bus",  type=int, default=cfg.IMU_BUS)
    parser.add_argument("--addr", type=lambda x: int(x, 0), default=cfg.IMU_ADDRESS)
    args = parser.parse_args()
    test_imu(args.duration, args.bus, args.addr)
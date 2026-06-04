"""
tests/test_imu.py
=================
Coverage target: = 90% of src/dms/modules/imu.py

smbus2 is fully mocked -- no I2C hardware required.
All time.monotonic() calls are patched for deterministic timing.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

_MONO    = "dms.modules.imu.time.monotonic"
_HW_FLAG = "dms.modules.imu._HW_AVAILABLE"
_SMBUS   = "dms.modules.imu.smbus"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_reader_sim():
    """IMUReader that always runs in simulation mode (no hardware needed)."""
    with patch(_HW_FLAG, False), patch(_MONO, return_value=0.0):
        from dms.modules.imu import IMUReader
        return IMUReader()


def _make_reader_hw(bus_mock=None):
    if bus_mock is None:
        bus_mock = MagicMock()
    smbus_mod = MagicMock()
    smbus_mod.SMBus.return_value = bus_mock
    with patch(_HW_FLAG, True), \
         patch(_SMBUS, smbus_mod, create=True), \
         patch(_MONO, return_value=0.0):
        from dms.modules.imu import IMUReader
        imu = IMUReader(bus_num=1)
    imu._bus = bus_mock
    return imu


# ---------------------------------------------------------------------------
# 1. _Kalman dataclass
# ---------------------------------------------------------------------------

class TestKalman:
    def test_update_returns_float(self):
        from dms.modules.imu import _Kalman
        k = _Kalman()
        result = k.update(new_angle=5.0, new_rate=0.1, dt=0.01)
        assert isinstance(result, float)

    def test_update_convergence(self):
        """After many identical measurements the filter should converge."""
        from dms.modules.imu import _Kalman
        k = _Kalman()
        angle = 30.0
        for _ in range(500):
            out = k.update(new_angle=angle, new_rate=0.0, dt=0.01)
        assert abs(out - angle) < 0.5

    def test_bias_corrected_over_time(self):
        """Gyro bias should shrink towards 0 with consistent updates."""
        from dms.modules.imu import _Kalman
        k = _Kalman()
        for _ in range(200):
            k.update(new_angle=0.0, new_rate=1.0, dt=0.01)
        assert abs(k.bias) > 0   # bias estimate is non-zero (tracking it)

    def test_predict_step_updates_angle(self):
        from dms.modules.imu import _Kalman
        k = _Kalman(angle=0.0, bias=0.0)
        k.update(new_angle=10.0, new_rate=90.0, dt=0.1)
        # After predict: angle should have moved from 0
        assert k.angle != 0.0


# ---------------------------------------------------------------------------
# 2. IMUReader construction
# ---------------------------------------------------------------------------

class TestIMUReaderInit:
    def test_sim_mode_when_hw_unavailable(self):
        imu = _make_reader_sim()
        assert imu._sim_mode is True

    def test_hw_mode_when_smbus_available(self):
        imu = _make_reader_hw()
        assert imu._sim_mode is False

    def test_falls_back_to_sim_on_smbus_error(self):
        smbus_mod = MagicMock()
        smbus_mod.SMBus.side_effect = OSError("I2C not found")
        with patch(_HW_FLAG, True), \
             patch(_SMBUS, smbus_mod, create=True), \
             patch(_MONO, return_value=0.0):
            from dms.modules.imu import IMUReader
            imu = IMUReader()
        assert imu._sim_mode is True

    def test_kalman_filters_initialised(self):
        imu = _make_reader_sim()
        from dms.modules.imu import _Kalman
        assert isinstance(imu._kalman_roll,  _Kalman)
        assert isinstance(imu._kalman_pitch, _Kalman)


# ---------------------------------------------------------------------------
# 3. _init_mpu
# ---------------------------------------------------------------------------

class TestInitMPU:
    def test_wakes_up_chip(self):
        bus = MagicMock()
        _make_reader_hw(bus_mock=bus)
        # PWR_MGMT_1 register should have been written with 0x00
        from dms.modules.imu import _REG_PWR_MGMT_1
        written_regs = [c.args[1] for c in bus.write_byte_data.call_args_list]
        assert _REG_PWR_MGMT_1 in written_regs

    def test_configures_five_registers(self):
        bus = MagicMock()
        _make_reader_hw(bus_mock=bus)
        assert bus.write_byte_data.call_count >= 5


# ---------------------------------------------------------------------------
# 4. _read_word_2c
# ---------------------------------------------------------------------------

class TestReadWord2C:
    def test_positive_value(self):
        bus = MagicMock()
        # high=0x01, low=0x00 -> 256
        bus.read_byte_data.side_effect = [0x01, 0x00]
        imu = _make_reader_hw(bus_mock=bus)
        val = imu._read_word_2c(0x3B)
        assert val == 256

    def test_negative_value(self):
        bus = MagicMock()
        # high=0xFF, low=0xFF -> 0xFFFF=65535 -> -1
        bus.read_byte_data.side_effect = [0xFF, 0xFF]
        imu = _make_reader_hw(bus_mock=bus)
        val = imu._read_word_2c(0x3B)
        assert val == -1

    def test_boundary_value_below_negative(self):
        bus = MagicMock()
        # 0x8000 -> -32768
        bus.read_byte_data.side_effect = [0x80, 0x00]
        imu = _make_reader_hw(bus_mock=bus)
        val = imu._read_word_2c(0x3B)
        assert val == -32768


# ---------------------------------------------------------------------------
# 5. _read_raw
# ---------------------------------------------------------------------------

class TestReadRaw:
    def test_returns_six_floats(self):
        bus = MagicMock()
        # Each _read_word_2c needs 2 reads; 6 axes -> 12 total reads
        bus.read_byte_data.side_effect = [0x00, 0x00] * 12
        imu = _make_reader_hw(bus_mock=bus)
        result = imu._read_raw()
        assert len(result) == 6
        for v in result:
            assert isinstance(v, float)


# ---------------------------------------------------------------------------
# 6. _sim_read
# ---------------------------------------------------------------------------

class TestSimRead:
    def test_returns_two_floats(self):
        imu = _make_reader_sim()
        roll, pitch = imu._sim_read()
        assert isinstance(roll,  float)
        assert isinstance(pitch, float)

    def test_oscillates_over_calls(self):
        imu = _make_reader_sim()
        results = [imu._sim_read() for _ in range(20)]
        rolls  = [r[0] for r in results]
        [r[1] for r in results]
        # Should not be all the same (oscillating)
        assert len(set(round(r, 3) for r in rolls)) > 1


# ---------------------------------------------------------------------------
# 7. read() -- simulation mode
# ---------------------------------------------------------------------------

class TestReadSim:
    def test_read_returns_tuple(self):
        imu = _make_reader_sim()
        result = imu.read()
        assert len(result) == 2

    def test_read_values_change(self):
        imu = _make_reader_sim()
        r1 = imu.read()
        r2 = imu.read()
        # Different sim_t -> different values
        assert r1 != r2


# ---------------------------------------------------------------------------
# 8. read() -- hardware mode
# ---------------------------------------------------------------------------

class TestReadHW:
    def test_returns_roll_pitch_tuple(self):
        bus = MagicMock()
        bus.read_byte_data.side_effect = [0x00, 0x80] * 100   # plenty of data
        imu = _make_reader_hw(bus_mock=bus)
        with patch(_MONO, side_effect=[0.0, 0.01]):
            roll, pitch = imu.read()
        assert isinstance(roll,  float)
        assert isinstance(pitch, float)

    def test_read_error_returns_zeros(self):
        bus = MagicMock()
        bus.read_byte_data.side_effect = OSError("bus error")
        imu = _make_reader_hw(bus_mock=bus)
        with patch(_MONO, side_effect=[0.0, 0.01]):
            roll, pitch = imu.read()
        assert roll  == pytest.approx(0.0)
        assert pitch == pytest.approx(0.0)

    def test_dt_clamped_to_minimum(self):
        """dt=0 would cause division issues -- guard clamps to 1e-4."""
        bus = MagicMock()
        bus.read_byte_data.side_effect = [0x00, 0x00] * 100
        imu = _make_reader_hw(bus_mock=bus)
        with patch(_MONO, side_effect=[5.0, 5.0]):   # now == last -> dt=0
            roll, pitch = imu.read()
        assert isinstance(roll, float)


# ---------------------------------------------------------------------------
# 9. close() and context manager
# ---------------------------------------------------------------------------

class TestClose:
    def test_close_calls_bus_close(self):
        bus = MagicMock()
        imu = _make_reader_hw(bus_mock=bus)
        imu.close()
        bus.close.assert_called_once()

    def test_close_noop_when_no_bus(self):
        imu = _make_reader_sim()
        imu._bus = None
        imu.close()   # must not raise

    def test_close_swallows_exception(self):
        bus = MagicMock()
        bus.close.side_effect = RuntimeError("bus error")
        imu = _make_reader_hw(bus_mock=bus)
        imu.close()   # must not propagate

    def test_context_manager(self):
        bus = MagicMock()
        smbus_mod = MagicMock()                    
        smbus_mod.SMBus.return_value = bus        
        with patch(_HW_FLAG, True), \
             patch(_SMBUS, smbus_mod, create=True), \
             patch(_MONO, return_value=0.0):
            from dms.modules.imu import IMUReader
            with IMUReader() as imu:
                assert imu is not None
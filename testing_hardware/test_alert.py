# tests/test_alert.py
"""
Interactive actuator test for all DMS hardware outputs.

Tests:
  1. Green  LED  (Pin 7 / GPIO12)
  2. Yellow LED  (Pin 29 / GPIO01)
  3. Red    LED  (Pin 31 / GPIO11)
  4. Vibration motor (Pin 33 / GPIO13 via 2N7000 MOSFET)
  5. USB sound buzzer (plays a tone via ALSA / aplay)
  6. Full alert sequence (NORMAL -> WARNING -> CRITICAL)

Run from project root:
    pdm run python tests/test_alert.py

Requirements:
    Jetson.GPIO   (pip install Jetson.GPIO)
    ALSA utils    (sudo apt-get install alsa-utils)  -- for aplay buzzer test
"""

import os
import sys
import time
import subprocess
import tempfile
import struct
import wave

# ---------------------------------------------------------------------------
# Path setup so we can import dms.config without pdm run
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import dms.config as cfg

# ---------------------------------------------------------------------------
# Colour helpers for terminal output
# ---------------------------------------------------------------------------
_GRN  = "\033[92m"
_YLW  = "\033[93m"
_RED  = "\033[91m"
_CYN  = "\033[96m"
_RST  = "\033[0m"
_BOLD = "\033[1m"


def _ok(msg):   print("{}  [PASS]  {}{}".format(_GRN,  msg, _RST))
def _fail(msg): print("{}  [FAIL]  {}{}".format(_RED,  msg, _RST))
def _info(msg): print("{}  [INFO]  {}{}".format(_CYN,  msg, _RST))
def _hdr(msg):  print("\n{}{}{}\n{}{}{}".format(
                    _BOLD, msg, _RST, _BOLD, "-" * len(msg), _RST))


def _ask(prompt):
    """Ask the user y/n and return True for yes."""
    while True:
        ans = input("{}  --> {}  [y/n]: ".format(_YLW, prompt) + _RST).strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False


# ---------------------------------------------------------------------------
# GPIO setup
# ---------------------------------------------------------------------------

def _setup_gpio():
    """Import Jetson.GPIO, set BOARD mode, configure output pins."""
    try:
        import Jetson.GPIO as GPIO
        GPIO.setmode(GPIO.BOARD)
        GPIO.setwarnings(False)
        pins = [
            cfg.ALERT_GPIO_GREEN,
            cfg.ALERT_GPIO_YELLOW,
            cfg.ALERT_GPIO_RED,
            cfg.ALERT_GPIO_VIB,
        ]
        GPIO.setup(pins, GPIO.OUT, initial=GPIO.LOW)
        _ok("GPIO initialised  (BOARD mode)  pins={}".format(pins))
        return GPIO
    except ImportError:
        _fail("Jetson.GPIO not found -- install with: pip install Jetson.GPIO")
        return None
    except Exception as exc:
        _fail("GPIO setup failed: {}".format(exc))
        return None


def _cleanup_gpio(GPIO):
    if GPIO:
        try:
            GPIO.cleanup()
            _info("GPIO cleaned up")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Individual actuator tests
# ---------------------------------------------------------------------------

def test_led(GPIO, pin, colour_name, colour_code):
    _hdr("TEST: {} LED  (BOARD pin {})".format(colour_name, pin))
    if GPIO is None:
        _fail("GPIO not available -- skipping")
        return False

    try:
        _info("Turning {} LED ON for 2 seconds...".format(colour_name))
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(2.0)
        GPIO.output(pin, GPIO.LOW)
        _info("{} LED OFF".format(colour_name))
    except Exception as exc:
        _fail("LED control error: {}".format(exc))
        return False

    passed = _ask("{}Did the {} LED light up?{}".format(colour_code, colour_name, _RST))
    if passed:
        _ok("{} LED working".format(colour_name))
    else:
        _fail("{} LED NOT working -- check wiring on pin {}".format(colour_name, pin))
    return passed


def test_vibration(GPIO, pin):
    _hdr("TEST: Vibration Motor  (BOARD pin {})".format(pin))
    if GPIO is None:
        _fail("GPIO not available -- skipping")
        return False

    try:
        _info("Vibration ON for 1 second...")
        GPIO.output(pin, GPIO.HIGH)
        time.sleep(1.0)
        GPIO.output(pin, GPIO.LOW)
        _info("Vibration OFF  (0.5 s pause)")
        time.sleep(0.5)

        _info("Vibration pulse pattern: 3 x 200 ms...")
        for i in range(3):
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(0.2)
            GPIO.output(pin, GPIO.LOW)
            time.sleep(0.2)
    except Exception as exc:
        _fail("Vibration motor control error: {}".format(exc))
        return False

    passed = _ask("Did the vibration motor activate?")
    if passed:
        _ok("Vibration motor working")
    else:
        _fail("Vibration motor NOT working -- check MOSFET wiring on pin {}".format(pin))
    return passed


def _make_wav(frequency=1000, duration=0.5, sample_rate=44100, amplitude=28000):
    """
    Generate a pure sine-wave WAV file in a temp file.
    Returns the file path (caller must delete it).
    frequency  Hz   -- 1000 Hz = standard test tone
    duration   s
    amplitude  0-32767
    """
    import math
    n_samples = int(sample_rate * duration)
    wav_path  = tempfile.mktemp(suffix=".wav")

    with wave.open(wav_path, "w") as wf:
        wf.setnchannels(1)       # mono
        wf.setsampwidth(2)       # 16-bit
        wf.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_samples):
            val = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
            frames += struct.pack("<h", val)
        wf.writeframes(bytes(frames))

    return wav_path


def _play_wav(wav_path, label):
    """Try aplay first, then paplay, then sox play."""
    for cmd in (
        ["aplay",    "-q", wav_path],
        ["paplay",         wav_path],
        ["play",     "-q", wav_path],
    ):
        try:
            result = subprocess.run(cmd, timeout=5,
                                    stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            if result.returncode == 0:
                _info("Played {} via '{}'".format(label, cmd[0]))
                return True
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            _fail("Audio playback timed out")
            return False

    _fail("No audio player found.  Install one:\n"
          "       sudo apt-get install alsa-utils\n"
          "       sudo apt-get install sox")
    return False


def test_buzzer():
    _hdr("TEST: USB Sound Buzzer")

    # List ALSA playback devices so the user can see what's detected
    try:
        result = subprocess.run(["aplay", "-l"], capture_output=True, text=True, timeout=3)
        _info("ALSA playback devices detected:")
        for line in result.stdout.splitlines():
            if line.strip():
                print("         {}".format(line))
    except FileNotFoundError:
        _info("aplay not found -- install alsa-utils for device listing")
    except Exception:
        pass

    tones = [
        (440,  0.4, "440 Hz  (A4 -- low beep)"),
        (1000, 0.4, "1000 Hz (standard alert tone)"),
        (2000, 0.4, "2000 Hz (high-pitched alert)"),
    ]

    results = []
    for freq, dur, label in tones:
        _info("Playing {}...".format(label))
        wav = _make_wav(frequency=freq, duration=dur)
        try:
            ok = _play_wav(wav, label)
            results.append(ok)
        finally:
            try:
                os.remove(wav)
            except Exception:
                pass
        time.sleep(0.3)

    if not any(results):
        _fail("No tones played successfully")
        _info("Troubleshooting:\n"
              "       1. Check USB sound card is plugged in\n"
              "       2. Run: aplay -l    (should list a USB audio device)\n"
              "       3. Run: amixer      (check volume is not 0)\n"
              "       4. Run: sudo apt-get install alsa-utils")
        passed = False
    else:
        passed = _ask("Did you hear the buzzer tones?")

    if passed:
        _ok("USB sound buzzer working")
    else:
        _fail("USB sound buzzer NOT working -- see troubleshooting above")
    return passed


# ---------------------------------------------------------------------------
# Full alert sequence test
# ---------------------------------------------------------------------------

def test_full_sequence(GPIO):
    _hdr("TEST: Full Alert Sequence  (NORMAL -> WARNING -> CRITICAL -> NORMAL)")
    if GPIO is None:
        _fail("GPIO not available -- skipping sequence test")
        return False

    _info("NORMAL state -- Green LED ON  (2 s)")
    GPIO.output(cfg.ALERT_GPIO_GREEN,  GPIO.HIGH)
    GPIO.output(cfg.ALERT_GPIO_YELLOW, GPIO.LOW)
    GPIO.output(cfg.ALERT_GPIO_RED,    GPIO.LOW)
    GPIO.output(cfg.ALERT_GPIO_VIB,    GPIO.LOW)
    time.sleep(2.0)

    _info("WARNING state -- Yellow LED ON + short vibration pulse  (2 s)")
    GPIO.output(cfg.ALERT_GPIO_GREEN,  GPIO.LOW)
    GPIO.output(cfg.ALERT_GPIO_YELLOW, GPIO.HIGH)
    GPIO.output(cfg.ALERT_GPIO_VIB,    GPIO.HIGH)
    time.sleep(0.3)
    GPIO.output(cfg.ALERT_GPIO_VIB,    GPIO.LOW)

    wav = _make_wav(frequency=800, duration=0.3)
    try:
        _play_wav(wav, "warning beep")
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass

    time.sleep(1.5)

    _info("CRITICAL state -- Red LED ON + sustained vibration + alarm  (3 s)")
    GPIO.output(cfg.ALERT_GPIO_YELLOW, GPIO.LOW)
    GPIO.output(cfg.ALERT_GPIO_RED,    GPIO.HIGH)
    GPIO.output(cfg.ALERT_GPIO_VIB,    GPIO.HIGH)

    wav = _make_wav(frequency=2000, duration=1.5, amplitude=30000)
    try:
        _play_wav(wav, "critical alarm")
    finally:
        try:
            os.remove(wav)
        except Exception:
            pass

    time.sleep(1.5)

    _info("Back to NORMAL -- all outputs OFF")
    GPIO.output(cfg.ALERT_GPIO_GREEN,  GPIO.LOW)
    GPIO.output(cfg.ALERT_GPIO_YELLOW, GPIO.LOW)
    GPIO.output(cfg.ALERT_GPIO_RED,    GPIO.LOW)
    GPIO.output(cfg.ALERT_GPIO_VIB,    GPIO.LOW)

    passed = _ask("Did the full sequence look and sound correct?")
    if passed:
        _ok("Full alert sequence working")
    else:
        _fail("Full sequence had issues -- check individual test results above")
    return passed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n{}{}=== DMS Actuator Test ==={}\n".format(_BOLD, _CYN, _RST))
    _info("Config pins:")
    _info("  Green  LED      = BOARD pin {}".format(cfg.ALERT_GPIO_GREEN))
    _info("  Yellow LED      = BOARD pin {}".format(cfg.ALERT_GPIO_YELLOW))
    _info("  Red    LED      = BOARD pin {}".format(cfg.ALERT_GPIO_RED))
    _info("  Vibration motor = BOARD pin {}".format(cfg.ALERT_GPIO_VIB))
    _info("  Buzzer          = USB sound card (ALSA)")
    print()

    GPIO = _setup_gpio()

    results = {}

    results["Green LED"]   = test_led(GPIO, cfg.ALERT_GPIO_GREEN,  "Green",  _GRN)
    results["Yellow LED"]  = test_led(GPIO, cfg.ALERT_GPIO_YELLOW, "Yellow", _YLW)
    results-["Red LED"]     = test_led(GPIO, cfg.ALERT_GPIO_RED,    "Red",    _RED)
    results["Vibration"]   = test_vibration(GPIO, cfg.ALERT_GPIO_VIB)
    results["Buzzer"]      = test_buzzer()
    results["Sequence"]    = test_full_sequence(GPIO)

    _cleanup_gpio(GPIO)

    # Summary
    _hdr("SUMMARY")
    all_pass = True
    for name, passed in results.items():
        if passed:
            _ok(name)
        else:
            _fail(name)
            all_pass = False

    print()
    if all_pass:
        print("{}{}All actuators PASSED{}\n".format(_BOLD, _GRN, _RST))
    else:
        failed = [n for n, p in results.items() if not p]
        print("{}{}FAILED: {}{}\n".format(_BOLD, _RED, ", ".join(failed), _RST))

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
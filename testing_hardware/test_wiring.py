#!/usr/bin/env python3
# tests/test_wiring.py
"""
DMS hardware wiring diagnostic v3.
Fixes in this version:
  - Correct sysfs GPIO base: gpiochip0 starts at 348 on Orin Nano Super JP6
  - GPIO numbers recomputed from Tegra234 port map (PH, PQ ports)
  - Auto-discovers real GPIO numbers using gpioinfo if gpiod is installed
  - Jetson.GPIO library crash bypassed (known JP6 model-detection bug)
  - USB audio forced to card 0 (USB2.0 Device) not the default HDMI card
  - speaker-test uses -D hw:0,0 to hit the correct USB card

Run as root:
    sudo python3 tests/test_wiring.py
"""

import os
import sys
import time
import subprocess
import glob
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

import dms.config as cfg

_GRN  = "\033[92m"
_YLW  = "\033[93m"
_RED  = "\033[91m"
_CYN  = "\033[96m"
_MAG  = "\033[95m"
_RST  = "\033[0m"
_BOLD = "\033[1m"

def _ok(m):   print("{}  [PASS]  {}{}".format(_GRN, m, _RST))
def _fail(m): print("{}  [FAIL]  {}{}".format(_RED, m, _RST))
def _warn(m): print("{}  [WARN]  {}{}".format(_YLW, m, _RST))
def _info(m): print("{}  [INFO]  {}{}".format(_CYN, m, _RST))
def _hdr(m):  print("\n{}{}{}\n{}{}{}".format(_BOLD, m, _RST, _BOLD, "-"*len(m), _RST))
def _tip(m):  print("{}  [TIP ]  {}{}".format(_MAG, m, _RST))


# ---------------------------------------------------------------------------
# GPIO number discovery
#
# Tegra234 (Jetson Orin Nano Super) gpiochip0 base = 348
# Port sizes (from kernel output you provided):
#   PA=8, PB=1, PC=8, PD=4, PE=5 ... PH=8, PI=8 ... PQ=8 ...
#
# Full port offset table (cumulative, base=348):
#   PA: 348-355  (8 pins, offset 0)
#   PB: 356      (1 pin,  offset 8)
#   PC: 357-364  (8 pins, offset 9)
#   PD: 365-368  (4 pins, offset 17)
#   PE: 369-373  (5 pins, offset 21)
#   PF: 374-377  (4 pins, offset 26)  -- estimated
#   PG: 378-383  (6 pins, offset 30)  -- estimated
#   PH: 384-391  (8 pins, offset 36)  <-- KEY: PH.00=384, PH.06=390
#   PI: 392-399  (8 pins, offset 44)
#   PJ: not present on this package
#   PK: 400-407? varies
#   ...
#   PQ: varies -- need gpioinfo to confirm
#
# CONFIRMED values from Jetson Orin Nano community (JP6.0/6.1):
# ---------------------------------------------------------------------------
_PIN_MAP = {
    #  BOARD: (gpio_num,  port_name,  label)
    13: (390, "PH.06", "GPIO12 -- Green LED"),
    29: (481, "PQ.05", "GPIO01 -- Yellow LED"),
    31: (482, "PQ.06", "GPIO11 -- Red LED"),
    33: (384, "PH.00", "GPIO13 -- Vibration motor"),
}

# Will be updated by auto-discovery if gpioinfo is available
_discovered = {}


def _run(cmd, timeout=5):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout + r.stderr, r.returncode
    except FileNotFoundError:
        return "", -1
    except subprocess.TimeoutExpired:
        return "", -2


# ---------------------------------------------------------------------------
# Auto-discover GPIO numbers using gpioinfo (most reliable method)
# ---------------------------------------------------------------------------

def auto_discover_gpio():
    _hdr("AUTO-DISCOVERY: Finding real GPIO numbers via gpioinfo")

    out, rc = _run(["gpioinfo", "gpiochip0"])
    if rc != 0:
        _warn("gpioinfo not available.  Install with:  sudo apt-get install -y gpiod")
        _warn("Using estimated GPIO numbers -- they may be wrong.")
        _tip("Run:  sudo apt-get install -y gpiod  then re-run this script")
        return False

    # gpioinfo output format:
    #   line  62: "PH.06"  unnamed  input  active-high
    port_to_gpio = {}
    for line in out.splitlines():
        m = re.match(r'\s+line\s+(\d+):\s+"([^"]+)"', line)
        if m:
            line_num = int(m.group(1))
            port_name = m.group(2)
            # gpiochip0 base is 348
            gpio_num = 348 + line_num
            port_to_gpio[port_name] = gpio_num

    if not port_to_gpio:
        _warn("gpioinfo ran but could not parse output")
        return False

    _ok("gpioinfo parsed {} lines from gpiochip0".format(len(port_to_gpio)))

    updated = 0
    for board_pin, (est_gpio, port_name, label) in _PIN_MAP.items():
        if port_name in port_to_gpio:
            real_gpio = port_to_gpio[port_name]
            _discovered[board_pin] = real_gpio
            if real_gpio != est_gpio:
                _warn("BOARD pin {}: {} -> estimated {} but real is {}".format(
                      board_pin, port_name, est_gpio, real_gpio))
                _PIN_MAP[board_pin] = (real_gpio, port_name, label)
            else:
                _ok("BOARD pin {}: {} = GPIO {} (confirmed)".format(
                    board_pin, port_name, real_gpio))
            updated += 1
        else:
            _warn("Port {} not found in gpioinfo output".format(port_name))

    if updated < len(_PIN_MAP):
        _warn("Could not confirm all pins.  Check gpioinfo output manually:")
        _tip("  gpioinfo gpiochip0 | grep -E 'PH\\.|PQ\\.'")

    # Also check gpiochip1 for AON pins
    out1, rc1 = _run(["gpioinfo", "gpiochip1"])
    if rc1 == 0:
        _info("gpiochip1 (AON) lines:")
        for line in out1.splitlines()[:10]:
            if line.strip():
                print("    {}".format(line))

    return updated > 0


# ---------------------------------------------------------------------------
# Sysfs GPIO helpers
# ---------------------------------------------------------------------------

def _export(gpio_num):
    path = "/sys/class/gpio/gpio{}".format(gpio_num)
    if not os.path.exists(path):
        try:
            with open("/sys/class/gpio/export", "w") as f:
                f.write(str(gpio_num))
            time.sleep(0.15)
        except Exception as e:
            return False, str(e)
    return os.path.exists(path), "ok"


def _unexport(gpio_num):
    try:
        with open("/sys/class/gpio/unexport", "w") as f:
            f.write(str(gpio_num))
    except Exception:
        pass


def _direction(gpio_num, d="out"):
    try:
        with open("/sys/class/gpio/gpio{}/direction".format(gpio_num), "w") as f:
            f.write(d)
        return True
    except Exception:
        return False


def _write(gpio_num, val):
    try:
        with open("/sys/class/gpio/gpio{}/value".format(gpio_num), "w") as f:
            f.write(str(val))
        return True
    except Exception:
        return False


def _read(gpio_num):
    try:
        with open("/sys/class/gpio/gpio{}/value".format(gpio_num)) as f:
            return int(f.read().strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GPIO pin test
# ---------------------------------------------------------------------------

def test_gpio_pin(label, board_pin):
    gpio_num, port_name, _ = _PIN_MAP[board_pin]
    _hdr("TEST: {}  (BOARD pin {} -> {} -> GPIO {})".format(
         label, board_pin, port_name, gpio_num))

    # Export
    ok, msg = _export(gpio_num)
    if not ok:
        _fail("Cannot export GPIO {} -- {}".format(gpio_num, msg))
        _tip("This means GPIO {} does not exist on this kernel.".format(gpio_num))
        _tip("The GPIO number for port {} may differ on your JetPack version.".format(port_name))
        _tip("Run:  gpioinfo gpiochip0 | grep {}".format(port_name))
        return False

    _ok("Exported GPIO {} ({})".format(gpio_num, port_name))

    # Set output direction
    if not _direction(gpio_num, "out"):
        _fail("Cannot set direction -- pin may be owned by another driver")
        _tip("Check:  sudo cat /sys/kernel/debug/gpio | grep {}".format(gpio_num))
        _unexport(gpio_num)
        return False

    _ok("Direction = OUTPUT")

    # Drive HIGH -- hold for 3s so user can measure with multimeter
    _write(gpio_num, 1)
    rb = _read(gpio_num)
    if rb == 1:
        _ok("Kernel confirms: pin is HIGH (3.3V should be present)")
    else:
        _fail("Kernel readback = {} after writing HIGH -- pin may be stuck".format(rb))
        _tip("Possible causes: wrong GPIO number, pin muxed to a peripheral,")
        _tip("  or pin is an input-only pin on this package.")

    print()
    print("{}  MULTIMETER CHECK -- {} (BOARD pin {}):{}".format(_YLW, label, board_pin, _RST))
    print("    Black probe: any GND pin  (BOARD pin 6, 9, 14, 20, 25, 30, 34, or 39)")
    print("    Red   probe: BOARD pin {}  (your signal wire)".format(board_pin))
    print("    Expected:    ~3.3 V")
    if "LED" in label:
        print("    At LED anode (after resistor): still ~3.3 V when connected")
        print("    Across LED:  ~1.8 to 2.2 V (voltage drop)")
        print("    If 0 V at pin: GPIO number still wrong, or no connection")
        print("    If 3.3 V at pin but LED off: check polarity, check resistor (220-330 ohm)")
    elif "Vibration" in label or "Motor" in label:
        print("    At MOSFET Gate (pin 2 of 2N7000): ~3.3 V")
        print("    At MOSFET Drain (pin 3): ~0 V when motor running (FET conducting)")
        print("    Motor supply: measure between motor + and GND: should be ~5 V")
        print("    If Gate = 3.3 V but motor off: check 5V supply on motor +")
        print("    2N7000 flat face toward you: left=Source, middle=Gate, right=Drain")
    print("    Holding HIGH for 3 seconds...")
    print()

    time.sleep(3.0)

    # Drive LOW
    _write(gpio_num, 0)
    rb = _read(gpio_num)
    if rb == 0:
        _ok("Kernel confirms: pin is LOW (0 V)")
    else:
        _warn("Readback = {} after writing LOW".format(rb))

    # Blink 5 times
    _info("Blinking 5x so you can see it with your eye / multimeter...")
    for _ in range(5):
        _write(gpio_num, 1)
        time.sleep(0.4)
        _write(gpio_num, 0)
        time.sleep(0.4)

    _unexport(gpio_num)

    ans = input("\n{}  --> Did {} respond? [y/n]: {}".format(
                _YLW, label, _RST)).strip().lower()
    if ans in ("y", "yes"):
        _ok("{} WORKING".format(label))
        return True
    else:
        _fail("{} NOT responding".format(label))
        return False


# ---------------------------------------------------------------------------
# USB audio test -- forced to card 0 (your USB2.0 Device)
# ---------------------------------------------------------------------------

def test_usb_audio():
    _hdr("TEST: USB Sound Buzzer  (ALSA card 0: USB2.0 Device)")

    _info("Your aplay -l shows:")
    _info("  card 0: USB2.0 Device  <-- this is your buzzer")
    _info("  card 1: NVIDIA HDA (HDMI -- not the buzzer)")
    print()

    # Check volume on card 0
    _info("Checking mixer levels on card 0...")
    out, rc = _run(["amixer", "-c", "0", "contents"])
    if rc == 0:
        lines = [l for l in out.splitlines() if "dB" in l or "%" in l]
        for l in lines[:8]:
            print("    {}".format(l.strip()))
    else:
        _warn("amixer not available or card 0 has no mixer controls")

    # Unmute and set volume
    _info("Setting card 0 volume to 90%...")
    for ctrl in ("Master", "Speaker", "PCM", "Headphone"):
        _run(["amixer", "-c", "0", "sset", ctrl, "90%", "unmute"])

    # Test 1: speaker-test on hw:0,0
    _info("Running speaker-test on hw:0,0 (your USB card) for 2 seconds...")
    try:
        proc = subprocess.Popen(
            ["speaker-test", "-D", "hw:0,0", "-t", "sine", "-f", "1000",
             "-c", "1", "-l", "1"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        time.sleep(2.5)
        proc.terminate()
        proc.wait(timeout=2)
        _ok("speaker-test ran without crash on hw:0,0")
    except FileNotFoundError:
        _fail("speaker-test not found -- run: sudo apt-get install alsa-utils")
        return False
    except Exception as e:
        _warn("speaker-test error: {}".format(e))

    # Test 2: aplay with a generated WAV
    _info("Generating and playing 1 kHz test tone via aplay -D hw:0,0 ...")
    import struct, wave, tempfile, math
    wav_path = tempfile.mktemp(suffix=".wav")
    sr, dur, freq = 44100, 1.0, 1000
    n = int(sr * dur)
    try:
        with wave.open(wav_path, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            frames = bytearray()
            for i in range(n):
                val = int(28000 * math.sin(2 * math.pi * freq * i / sr))
                frames += struct.pack("<h", val)
            wf.writeframes(bytes(frames))

        result = subprocess.run(
            ["aplay", "-D", "hw:0,0", "-q", wav_path],
            timeout=4, capture_output=True
        )
        if result.returncode == 0:
            _ok("aplay played 1 kHz tone on hw:0,0 without error")
        else:
            _warn("aplay returned error: {}".format(
                  result.stderr.decode(errors="ignore").strip()))
            _tip("Try:  aplay -D plughw:0,0 /tmp/test.wav  (plughw does rate conversion)")
    except Exception as e:
        _warn("WAV play error: {}".format(e))
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass

    # Test 3: try plughw in case hw:0,0 has sample rate mismatch
    _info("Also trying plughw:0,0 (handles sample rate conversion)...")
    try:
        import struct, wave, tempfile, math
        wav_path2 = tempfile.mktemp(suffix=".wav")
        with wave.open(wav_path2, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            frames = bytearray()
            for i in range(44100):
                val = int(28000 * math.sin(2 * math.pi * 1000 * i / 44100))
                frames += struct.pack("<h", val)
            wf.writeframes(bytes(frames))
        result = subprocess.run(
            ["aplay", "-D", "plughw:0,0", "-q", wav_path2],
            timeout=4, capture_output=True
        )
        if result.returncode == 0:
            _ok("plughw:0,0 playback succeeded")
        else:
            _warn("plughw:0,0 error: {}".format(result.stderr.decode(errors="ignore").strip()))
    except Exception as e:
        _warn("plughw test error: {}".format(e))
    finally:
        try:
            os.remove(wav_path2)
        except Exception:
            pass

    ans = input("\n{}  --> Did you hear a tone from the buzzer? [y/n]: {}".format(
                _YLW, _RST)).strip().lower()
    if ans in ("y", "yes"):
        _ok("USB buzzer WORKING")
        return True
    else:
        _fail("USB buzzer NOT working")
        print()
        print("  Troubleshooting steps:")
        print("  1. Physical: buzzer + wire to USB card output, - wire to GND")
        print("     Some USB cards have a 3.5mm jack -- try headphones first to")
        print("     confirm the card itself works before testing the buzzer.")
        print("  2. Check card is not muted:")
        print("     amixer -c 0")
        print("  3. Force specific device string:")
        print("     aplay -D plughw:0,0 /tmp/test.wav")
        print("  4. List all controls on card 0:")
        print("     amixer -c 0 contents")
        print("  5. Try with headphones on the USB card jack to isolate")
        print("     whether the card works vs buzzer wiring.")
        return False


# ---------------------------------------------------------------------------
# config.py update helper
# ---------------------------------------------------------------------------

def show_config_fix():
    _hdr("config.py -- CORRECTED GPIO NUMBERS")
    print()
    print("  Replace the GPIO pin constants in src/dms/config.py with:")
    print()
    for board_pin, (gpio_num, port_name, label) in sorted(_PIN_MAP.items()):
        print("  # {} (BOARD pin {}, {})".format(label, board_pin, port_name))
    print()
    print("  The BOARD pin numbers in config.py are CORRECT (15, 29, 31, 33).")
    print("  Jetson.GPIO uses BOARD mode so it maps BOARD pins itself.")
    print("  The sysfs numbers above are only needed if Jetson.GPIO stays broken.")
    print()
    _hdr("Jetson.GPIO model detection fix")
    print()
    print("  The error 'Could not determine Jetson model' means your")
    print("  /proc/device-tree/compatible does not contain a string that")
    print("  the installed Jetson.GPIO version recognises.")
    print()
    print("  Fix option 1 -- upgrade Jetson.GPIO:")
    print("    sudo pip3 install --upgrade Jetson.GPIO")
    print()
    print("  Fix option 2 -- check your compatible string:")
    print("    cat /proc/device-tree/compatible | tr '\\0' '\\n'")
    print()
    print("  Fix option 3 -- use the gpio sysfs directly (what this script does)")
    print("    Replace Jetson.GPIO calls in alert.py with sysfs writes.")
    print("    A drop-in replacement class is provided in tests/sysfs_gpio.py")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("\n{}{}=== DMS Wiring Diagnostic v3 ==={}\n".format(_BOLD, _CYN, _RST))

    if os.geteuid() != 0:
        _fail("Must run as root:  sudo python3 tests/test_wiring.py")
        sys.exit(1)
    _ok("Running as root")

    # Step 1: auto-discover real GPIO numbers
    auto_discover_gpio()

    print()
    _info("GPIO numbers that will be used:")
    for board_pin, (gpio_num, port_name, label) in sorted(_PIN_MAP.items()):
        src = "gpioinfo" if board_pin in _discovered else "estimated"
        print("    BOARD {:>2}  ->  GPIO {:>4}  ({})  [{}]  {}".format(
              board_pin, gpio_num, port_name, src, label))

    results = {}
    results["Green LED"]        = test_gpio_pin("Green LED",       13)
    results["Yellow LED"]       = test_gpio_pin("Yellow LED",      29)
    results["Red LED"]          = test_gpio_pin("Red LED",         31)
    results["Vibration Motor"]  = test_gpio_pin("Vibration Motor", 33)
    results["USB Buzzer"]       = test_usb_audio()

    show_config_fix()

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
        print("{}{}All hardware PASSED{}\n".format(_BOLD, _GRN, _RST))
    else:
        failed = [n for n, p in results.items() if not p]
        print("{}{}FAILED: {}{}\n".format(_BOLD, _RED, ", ".join(failed), _RST))

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
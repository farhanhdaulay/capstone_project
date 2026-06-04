"""
Linux Sysfs GPIO Verification Script
Sets a pin HIGH, reads the kernel to verify it is HIGH, and turns it OFF.
"""
import os
import time
import sys

# The absolute Linux kernel IDs for your Yahboom pins
HARDWARE_PINS = {
    "Green LED": 7,
    "Yellow LED": 29,
    "Red LED": 31,
    "Vibration Motor": 33
}

def write_file(path, value):
    """Helper to write values to the kernel file system."""
    with open(path, 'w') as f:
        f.write(str(value))

def read_file(path):
    """Helper to read values from the kernel file system."""
    with open(path, 'r') as f:
        return f.read().strip()

def test_and_verify_pin(name, pin_number):
    print(f"\n>>> [TESTING] {name} (Kernel ID: {pin_number})")
    
    gpio_path = f"/sys/class/gpio/gpio{pin_number}"
    
    try:
        # 1. Export the pin to make it available to the user space
        if not os.path.exists(gpio_path):
            write_file("/sys/class/gpio/export", pin_number)
            time.sleep(0.1) # Wait for the OS to create the file structure
        
        # 2. Set direction to Output
        write_file(f"{gpio_path}/direction", "out")
        
        # 3. Turn the pin ON
        print("Commanding pin to HIGH (1)...")
        write_file(f"{gpio_path}/value", "1")
        
        # 4. THE VERIFICATION: Read the state back from the kernel
        current_state = read_file(f"{gpio_path}/value")
        
        if current_state == "1":
            print("[SUCCESS] Kernel confirms pin is currently ON.")
        else:
            print(f"[FAIL] Kernel reports pin is state: {current_state}")
            
        time.sleep(2.0) # Hold it on so you can look at the breadboard
        
        # 5. Turn the pin OFF
        print("Commanding pin to LOW (0)...")
        write_file(f"{gpio_path}/value", "0")
        
    except PermissionError:
        print("[ERROR] Permission denied. You must run this script with 'sudo'.")
    except Exception as e:
        print(f"[ERROR] Failed to verify {name}: {e}")
    finally:
        # 6. Safety Cleanup: Unexport the pin
        if os.path.exists(gpio_path):
            write_file("/sys/class/gpio/unexport", pin_number)

def main():
    print("========================================")
    print("  DMS GPIO State Verification Test")
    print("========================================")
    
    for name, pin in HARDWARE_PINS.items():
        test_and_verify_pin(name, pin)
        
    print("\n[Complete] All pins tested and safely closed.")

if __name__ == '__main__':
    main()
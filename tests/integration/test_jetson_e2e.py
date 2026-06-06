#!/usr/bin/env python3
# Copyright (c) 2026 Kishore Sridhar & Farhan Hikmatullah Daulay
# Tatung University 14210 AI實務專題

"""tests/integration/test_jetson_e2e.py
End-to-End Inference Test on Jetson hardware (Log-Based Verification).
"""

import os
import subprocess
import time
import pytest

IMAGE = os.environ.get("IMAGE", "ghcr.io/farhanhdaulay/capstone_project:latest")
CONTAINER_NAME = "dms_integration_test"

@pytest.fixture(scope="module")
def inference_container():
    """Starts the Docker container with GPU access and cleans it up after."""
    # Kill any leftover containers first
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    
    # ← ADD THIS: release camera from any zombie processes
    subprocess.run(["sudo", "fuser", "-k", "/dev/video0"], capture_output=True)
    subprocess.run(["sudo", "fuser", "-k", "/dev/video1"], capture_output=True)
    time.sleep(2)  # give OS time to release the device

    print(f"\nStarting container: {IMAGE}")
    start_cmd = [
        "docker", "run", "-d",
        "--name", CONTAINER_NAME,
        "--runtime", "nvidia",
        "--privileged",
        "-v", "dms-models:/app/models",
        "-v", "/dev:/dev",
        "--device", "/dev/video0:/dev/video0",
        "--device", "/dev/video1:/dev/video1",
        "--device", "/dev/bus/usb",
        IMAGE,
        "python3", "src/dms/main.py", "--no-window"
    ]
    
    result = subprocess.run(start_cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Failed to start container: {result.stderr}"
    
    yield CONTAINER_NAME
    
    print("\nStopping and cleaning up container...")
    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    
    # ← Release camera after test too
    subprocess.run(["sudo", "fuser", "-k", "/dev/video0"], capture_output=True)
    subprocess.run(["sudo", "fuser", "-k", "/dev/video1"], capture_output=True)

def test_inference_starts_successfully(inference_container):
    """Asserts that the container loads the model and runs inference by checking logs."""
    success = False
    
    print("\nWaiting up to 10 minutes for TensorRT engine compile/load...")
    deadline = time.time() + 600.0  # 10 minute timeout
    
    while time.time() < deadline:
        # 1. Fail-Fast: Check if the container actually crashed
        status = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", CONTAINER_NAME], 
            capture_output=True, text=True
        )
        if status.stdout.strip() == "false":
            print("\n❌ Container crashed prematurely! Stopping test.")
            break

        # 2. Grab logs
        logs = subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, text=True)
        
        # 3. Combine stdout and stderr (Python logging defaults to stderr)
        combined_logs = logs.stdout + logs.stderr
        
        # 4. Check for success flags
        if ("FPS:" in combined_logs or
            "Detection" in combined_logs or
            "Loop running" in combined_logs or
            "Calibration complete" in combined_logs or
            "State:" in combined_logs):
            success = True
            print("\n✅ Success condition met in logs!")
            break
            
        time.sleep(5) # Poll every 5 seconds
        
    if not success:
        # Fetch final logs for debugging if it failed
        final_logs = subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, text=True)
        print(f"\nCONTAINER LOGS:\n{final_logs.stdout}\n{final_logs.stderr}")
        
    assert success, "Container failed to reach active inference loop within timeout or crashed."

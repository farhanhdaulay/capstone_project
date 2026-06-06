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
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    
    print(f"\nStarting container: {IMAGE}")
    start_cmd = [
        "docker", "run", "-d", 
        # REMOVED --rm so we can actually read the crash logs!
        "--name", CONTAINER_NAME,
        "--runtime", "nvidia",
        "-v", "dms-models:/opt/models", 
        "--device", "/dev/video0:/dev/video0",
        "--device", "/dev/video1:/dev/video1",
        IMAGE,
        # OVERRIDE THE DEFAULT COMMAND TO FORCE SOURCE 0
        "python3", "src/dms/main.py", "--no-window", "--source", "0"
    ]
    
    result = subprocess.run(start_cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Failed to start container: {result.stderr}"
    
    yield CONTAINER_NAME
    
    print("\nStopping and cleaning up container...")
    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)
    # Added rm -f here to manually clean up since we removed --rm
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)

def test_inference_starts_successfully(inference_container):
    """Asserts that the container loads the model and runs inference by checking logs."""
    success = False
    
    print("\nWaiting up to 10 minutes for TensorRT engine compile/load...")
    deadline = time.time() + 600.0  # 10 minute timeout
    
    while time.time() < deadline:
        logs = subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, text=True)
        
        # We are looking for a log line that proves your pipeline ran successfully.
        # Adjust "FPS:" or "Detection" to exactly match whatever your inference_node prints!
        if "FPS:" in logs.stdout or "Detection" in logs.stdout:
            success = True
            break
            
        time.sleep(5) # Poll every 5 seconds
        
    if not success:
        print(f"\nCONTAINER LOGS:\n{logs.stdout}\n{logs.stderr}")
        
    assert success, "Container failed to reach active inference loop within timeout."
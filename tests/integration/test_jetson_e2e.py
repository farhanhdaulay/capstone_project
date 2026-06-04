#!/usr/bin/env python3
# Copyright (c) 2026 Kishore Sridhar & Farhan Hikmatullah Daulay
# Tatung University 14210 AI實務專題

"""tests/integration/test_jetson_e2e.py
End-to-End Inference Test on Jetson hardware.
"""

import os
import subprocess
import time
import threading
import json
import pytest
import paho.mqtt.client as mqtt
from paho.mqtt.enums import CallbackAPIVersion

# The CI workflow passes the image tag via the IMAGE env var
IMAGE = os.environ.get("IMAGE", "ghcr.io/farhanhdaulay/capstone_project:latest")
CONTAINER_NAME = "dms_integration_test"
MQTT_TOPIC = "jetson/vision/detections"

@pytest.fixture(scope="module")
def inference_container():
    """Starts the Docker container with GPU access and cleans it up after."""
    # Ensure no old test container is lingering
    subprocess.run(["docker", "rm", "-f", CONTAINER_NAME], capture_output=True)
    
    print(f"\nStarting container: {IMAGE}")
    start_cmd = [
        "docker", "run", "-d", "--rm",
        "--name", CONTAINER_NAME,
        "--runtime", "nvidia",
        "-v", "lab12-models:/opt/models", # Cache TensorRT engine
        "-e", "MQTT_BROKER=172.17.0.1",   # Connect to host Mosquitto
        IMAGE
    ]
    
    result = subprocess.run(start_cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"Failed to start container: {result.stderr}"
    
    yield CONTAINER_NAME
    
    # Cleanup runs even if the test fails
    print("\nStopping container...")
    subprocess.run(["docker", "stop", CONTAINER_NAME], capture_output=True)

def test_inference_publishes_mqtt_within_window(inference_container):
    """Asserts that the container loads the model and publishes an MQTT message."""
    message_received = threading.Event()
    received_payload = {}

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            if "detections" in payload:
                received_payload.update(payload)
                message_received.set()
        except json.JSONDecodeError:
            pass

    # Setup MQTT Subscriber
    client = mqtt.Client(callback_api_version=CallbackAPIVersion.VERSION2)
    client.on_message = on_message
    client.connect("localhost", 1883, 60)
    client.subscribe(MQTT_TOPIC)
    client.loop_start()

    print("\nWaiting up to 10 minutes for TensorRT engine compile/load...")
    # HW6 requires up to 10 min wait for first-time TensorRT compilation
    success = message_received.wait(timeout=600.0) 
    
    client.loop_stop()
    client.disconnect()

    if not success:
        # If it failed, grab the docker logs so we can see why
        logs = subprocess.run(["docker", "logs", CONTAINER_NAME], capture_output=True, text=True)
        print(f"\nCONTAINER LOGS:\n{logs.stdout}\n{logs.stderr}")
        
    assert success, "No MQTT detection message received within the timeout window."
    assert "frame" in received_payload, "Payload missing required schema fields."

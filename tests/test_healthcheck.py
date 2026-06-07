"""
tests/test_healthcheck.py
=========================
Coverage target: = 90% of src/dms/healthcheck.py
"""
import json
import time
import urllib.request
import urllib.error
from unittest.mock import MagicMock, mock_open, patch

import pytest

from dms.healthcheck import HealthCheckServer, _current_power_mode, start_in_thread


# ---------------------------------------------------------------------------
# 1. Hardware Mocking (_current_power_mode)
# ---------------------------------------------------------------------------

def test_power_mode_from_file():
    """Test reading nvpmodel from the file system (Docker mount)."""
    mock_data = "NVPM INFO: ...\nPower Model Name: 15W_DESKTOP\n"
    with patch("builtins.open", mock_open(read_data=mock_data)):
        assert _current_power_mode() == "15W_DESKTOP"


def test_power_mode_from_cmd():
    """Test reading nvpmodel from subprocess fallback."""
    # Force the file open to fail, triggering the subprocess
    with patch("builtins.open", side_effect=FileNotFoundError):
        mock_run = MagicMock()
        mock_run.stdout = "Power Mode : MAXN\n"
        with patch("subprocess.run", return_value=mock_run):
            assert _current_power_mode() == "MAXN"


def test_power_mode_fallback():
    """Test when neither file nor command is available (e.g. Windows)."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _current_power_mode() == ""


# ---------------------------------------------------------------------------
# 2. HTTP Server Endpoint
# ---------------------------------------------------------------------------

def test_healthz_endpoint_success():
    """Ensure the server spins up and returns 200 OK with valid JSON."""
    # Use a high, non-standard port to avoid conflicts
    server = HealthCheckServer(port=8123)
    server.start_in_thread()
    time.sleep(0.1)  # Give the daemon thread a moment to bind the socket

    # We mock the power mode so it doesn't actually try to run Jetson commands during the HTTP request
    with patch("dms.healthcheck._current_power_mode", return_value="MOCKED_MODE"):
        req = urllib.request.Request("http://127.0.0.1:8123/healthz")
        with urllib.request.urlopen(req) as res:
            assert res.status == 200
            assert res.headers.get_content_type() == "application/json"
            
            data = json.loads(res.read().decode())
            assert data["status"] == "healthy"
            assert data["power_mode"] == "MOCKED_MODE"
            assert "model_version" in data


def test_healthz_endpoint_404():
    """Ensure invalid endpoints return a 404 Error."""
    server = HealthCheckServer(port=8124)
    server.start_in_thread()
    time.sleep(0.1)

    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen("http://127.0.0.1:8124/invalid_path")
    
    assert excinfo.value.code == 404


def test_module_wrapper_function():
    """Test the convenient start_in_thread() wrapper."""
    with patch("dms.healthcheck.HealthCheckServer") as mock_server_class:
        mock_instance = MagicMock()
        mock_server_class.return_value = mock_instance
        
        start_in_thread()
        
        mock_instance.start_in_thread.assert_called_once()
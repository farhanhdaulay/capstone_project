#!/usr/bin/env python3
# Copyright (c) 2026 Kishore Sridhar & Farhan Hikmatullah Daulay
# Tatung University 14210 AI實務專題
"""src/dms/healthcheck.py — /healthz HTTP endpoint for the DMS container."""
from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("HEALTHZ_PORT", "8000"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")


def _current_power_mode() -> str:
    """Read live nvpmodel state from mounted status file or binary."""
    # Try reading from bind-mounted status file first (works inside container)
    try:
        status = open("/var/lib/nvpmodel/status").read()
        for line in status.splitlines():
            if "Power Model Name" in line:
                return line.split(":")[-1].strip()
    except (FileNotFoundError, PermissionError):
        pass
    # Fallback: try nvpmodel binary
    try:
        out = subprocess.run(
            ["nvpmodel", "-q"],
            capture_output=True, text=True, timeout=2,
        )
        for line in out.stdout.splitlines():
            if "Power Mode" in line:
                return line.split(":", 1)[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return ""


class HealthCheckServer:
    """Minimal HTTP server exposing /healthz."""

    def __init__(self, port: int = PORT) -> None:
        self.port = port

    def start_in_thread(self) -> threading.Thread:
        """Start the healthz server on a daemon thread."""
        server = HTTPServer(("0.0.0.0", self.port), _Handler) # nosec B104 — intentional container-internal bind, not exposed to public network
        t = threading.Thread(
            target=server.serve_forever,
            daemon=True,
            name="healthz",
        )
        t.start()
        return t


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path != "/healthz":
            self.send_error(404)
            return
        body = json.dumps({
            "status": "healthy",
            "model_version": MODEL_VERSION,
            "power_mode": _current_power_mode(),
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def start_in_thread() -> threading.Thread:
    """Module-level convenience wrapper used by main()."""
    return HealthCheckServer().start_in_thread()

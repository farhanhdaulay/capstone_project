#!/usr/bin/env python3
# Copyright (c) 2026 Kishore Sridhar & Farhan Hikmatullah Daulay
# Tatung University 14210 AI實務專題
"""src/healthcheck.py — minimal /healthz endpoint for the DMS container.
Started as a background thread by main() so every container gets the
endpoint for free. Returns JSON with status, model_version, power_mode.
"""
from __future__ import annotations

import json
import os
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

PORT = int(os.environ.get("HEALTHZ_PORT", "8000"))
MODEL_VERSION = os.environ.get("MODEL_VERSION", "unknown")


def _current_power_mode() -> str:
    """Read live nvpmodel state. Returns empty string if unavailable."""
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
    """Minimal HTTP server exposing /healthz for deploy-side polling."""

    def __init__(self, port: int = PORT) -> None:
        self.port = port
        self._server: HTTPServer | None = None

    def start_in_thread(self) -> threading.Thread:
        """Start the healthz server on a daemon thread."""
        self._server = HTTPServer(("0.0.0.0", self.port), _Handler)
        t = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="healthz",
        )
        t.start()
        return t


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
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
        pass  # silence per-request stderr spam


def start_in_thread() -> threading.Thread:
    """Module-level convenience wrapper used by main()."""
    server = HealthCheckServer()
    return server.start_in_thread()

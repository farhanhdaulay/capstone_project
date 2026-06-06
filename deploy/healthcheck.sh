#!/usr/bin/env bash
# Copyright (c) 2026 Kishore Sridhar & Farhan Hikmatullah Daulay
# Tatung University 14210 AI實務專題
# deploy/healthcheck.sh — polls /healthz, requires 3 consecutive successes

set -euo pipefail

URL="${HEALTHZ_URL:-http://localhost:8000/healthz}"
DEADLINE=$((SECONDS + 60))
STREAK=0
NEEDED=3

while [ "$SECONDS" -lt "$DEADLINE" ]; do
    if body=$(curl -fsS --max-time 2 "$URL" 2>/dev/null) && \
        python3 -c "
import sys, json
d = json.loads('$body'.replace(\"'\", '\"'))
sys.exit(0 if d.get('status') == 'healthy' else 1
)" 2>/dev/null; then
        STREAK=$((STREAK + 1))
        echo "[healthcheck] OK ($STREAK/$NEEDED): $body"
        [ "$STREAK" -ge "$NEEDED" ] && exit 0
    else
        [ "$STREAK" -gt 0 ] && echo "[healthcheck] streak broken at $STREAK"
        STREAK=0
    fi
    sleep 2
done

echo "[healthcheck] FAILED — no $NEEDED consecutive successes in 60s" >&2
exit 1

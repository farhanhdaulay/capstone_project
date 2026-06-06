#!/usr/bin/env bash
# Copyright (c) 2026 Kishore Sridhar & Farhan Hikmatullah Daulay
# Tatung University 14210 AI實務專題
# deploy/healthcheck.sh — polls /healthz, requires 3 consecutive successes

set -euo pipefail

URL="${HEALTHZ_URL:-http://localhost:8000/healthz}"
DEADLINE=$((SECONDS + 120))  # ← increased to 120s to allow app startup
STREAK=0
NEEDED=3

# ← Wait for app to initialize before first poll
echo "[healthcheck] Waiting 20s for app to initialize..."
sleep 20

while [ "$SECONDS" -lt "$DEADLINE" ]; do
    HTTP_CODE=$(curl -o /tmp/healthz_body.json -w "%{http_code}" \
        -fsS --max-time 2 "$URL" 2>/dev/null || echo "000")

    if [ "$HTTP_CODE" = "200" ]; then
        BODY=$(cat /tmp/healthz_body.json 2>/dev/null || echo "{}")
        STREAK=$((STREAK + 1))
        echo "[healthcheck] OK ($STREAK/$NEEDED): $BODY"
        [ "$STREAK" -ge "$NEEDED" ] && exit 0
    else
        [ "$STREAK" -gt 0 ] && echo "[healthcheck] streak broken at $STREAK (HTTP $HTTP_CODE)"
        STREAK=0
        echo "[healthcheck] waiting... (HTTP $HTTP_CODE)"
    fi
    sleep 3
done

echo "[healthcheck] FAILED — no $NEEDED consecutive successes in 120s" >&2
exit 1

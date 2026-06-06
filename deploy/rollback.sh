#!/usr/bin/env bash
# Copyright (c) 2026 Kishore Sridhar & Farhan Hikmatullah Daulay
# Tatung University 14210 AI????
# deploy/rollback.sh — revert to the previous deployed tag in <30s

set -euo pipefail

STATE_DIR="${STATE_DIR:-/var/lib/capstone_project}"
COMPOSE_FILE="$(dirname "$0")/docker-compose.yml"

# 1. Read previous tag
if [ ! -f "$STATE_DIR/deployed.txt.history" ]; then
    echo "[rollback] ERROR: No previous tag found in history" >&2
    exit 1
fi

PREV_TAG=$(tail -1 "$STATE_DIR/deployed.txt.history")
CURR_TAG=$(cat "$STATE_DIR/deployed.txt" 2>/dev/null || echo "unknown")

echo "[rollback] Rolling back from $CURR_TAG ? $PREV_TAG"

# 2. Pull previous tag (no-op if cached locally)
export IMAGE_TAG="$PREV_TAG"
docker compose -f "$COMPOSE_FILE" pull || true  # use cache if auth expired

# 3. Restart with previous tag
docker compose -f "$COMPOSE_FILE" up -d --force-recreate

# 4. Verify health
if ! bash "$(dirname "$0")/healthcheck.sh"; then
    echo "[rollback] ERROR: Rollback target $PREV_TAG is also unhealthy!" >&2
    echo "[rollback] Manual intervention required." >&2
    exit 1
fi

# 5. Update state files
echo "$PREV_TAG" > "$STATE_DIR/deployed.txt"
# Remove the last line from history (we've rolled back to it)
head -n -1 "$STATE_DIR/deployed.txt.history" > "$STATE_DIR/deployed.txt.history.tmp"
mv "$STATE_DIR/deployed.txt.history.tmp" "$STATE_DIR/deployed.txt.history"

echo "[rollback] Successfully rolled back to $PREV_TAG"
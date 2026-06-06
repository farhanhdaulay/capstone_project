#!/usr/bin/env bash
# Copyright (c) 2026 Kishore Sridhar & Farhan Hikmatullah Daulay
# Tatung University 14210 AI????
# deploy/deploy.sh — pull tag, set nvpmodel, restart compose, healthcheck

set -euo pipefail

TAG="${1:?Usage: deploy.sh <vX.Y.Z or sha-XXXXXXX>}"
ENV="${DEPLOY_ENV:-production}"
STATE_DIR="${STATE_DIR:-/var/lib/capstone_project}"
COMPOSE_FILE="$(dirname "$0")/docker-compose.yml"

mkdir -p "$STATE_DIR"

# 1. Resolve power mode NAME ? numeric ID
MODE_NAME=$(python3 -c "
import json, sys
p = json.load(open('deploy/power_profile.json'))
print(p.get('$ENV', p['production']))
")

PAT="<[[:space:]]*POWER_MODEL[[:space:]]+ID=[0-9]+[[:space:]]+NAME=${MODE_NAME}[[:space:]]*>"
MODE_ID=$(grep -oE "$PAT" /etc/nvpmodel.conf \
    | grep -oE "ID=[0-9]+" | cut -d= -f2 | head -1)

if [ -z "$MODE_ID" ]; then
    echo "[deploy] ERROR: power mode '$MODE_NAME' not in /etc/nvpmodel.conf"
    grep -oE "<[[:space:]]*POWER_MODEL[[:space:]]+ID=[0-9]+[[:space:]]+NAME=[^>]+>" \
        /etc/nvpmodel.conf
    exit 1
fi

echo "[deploy] Setting nvpmodel to $MODE_NAME (ID=$MODE_ID) for env=$ENV"
sudo nvpmodel -m "$MODE_ID"
sudo jetson_clocks
sleep 2
echo "[deploy] Freeing port 8000..."
sudo fuser -k 8000/tcp || true
sleep 1

# 2. Save previous tag for rollback
if [ -f "$STATE_DIR/deployed.txt" ]; then
    PREV=$(cat "$STATE_DIR/deployed.txt")
    echo "$PREV" >> "$STATE_DIR/deployed.txt.history"
    echo "[deploy] Previous tag: $PREV (saved for rollback)"
else
    echo "[deploy] First deploy — initializing state"
    echo "$TAG" >> "$STATE_DIR/deployed.txt.history"
fi

# 3. Pull and restart
export IMAGE_TAG="$TAG"
docker compose -f "$COMPOSE_FILE" pull || \
    echo "[deploy] WARNING: pull failed, using local cache"
docker compose -f "$COMPOSE_FILE" up -d --force-recreate

echo "[deploy] Waiting 25s for app to initialize..."
sleep 25

# 4. Healthcheck — rollback on failure
if ! bash "$(dirname "$0")/healthcheck.sh"; then
    echo "[deploy] Healthcheck failed — rolling back"
    if [ -x "$(dirname "$0")/rollback.sh" ]; then
        bash "$(dirname "$0")/rollback.sh"
    else
        echo "[deploy] WARNING: rollback.sh not yet available"
    fi
    exit 1
fi

# 5. Record new deployed tag
echo "$TAG" > "$STATE_DIR/deployed.txt"
echo "[deploy] Successfully deployed $TAG at power mode $MODE_NAME"
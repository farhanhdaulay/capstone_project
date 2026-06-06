FROM dustynv/pytorch:2.7-r36.4.0

ENV DEBIAN_FRONTEND=noninteractive
# Override dustynv's private pip mirror to use public PyPI
ENV PIP_INDEX_URL=https://pypi.org/simple/
ENV PIP_EXTRA_INDEX_URL=

# 1. Install OS dependencies
RUN apt-get update -o Acquire::Retries=5 && \
    apt-get install -y --allow-unauthenticated --no-install-recommends -o Acquire::Retries=5 \
    libgl1-mesa-glx libglib2.0-0 alsa-utils libgomp1 libopenblas0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. COPY REQUIREMENTS FIRST (Crucial for caching)
COPY requirements.txt .

# 3. PIP INSTALL WITH BUILDKIT CACHE MOUNT 
# (This is the magic line that stops 4-hour rebuilds)
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --no-cache-dir -r requirements.txt

# 4. Install custom ONNXRuntime wheel
RUN --mount=type=cache,target=/root/.cache/pip \
    rm -rf /usr/local/lib/python3.10/dist-packages/onnxruntime* && \
    pip3 install --no-cache-dir https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

# 5. Stub ONNX so Ultralytics doesn't crash on startup
RUN ORT_VER=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)") && \
    STUB=/usr/local/lib/python3.10/dist-packages/onnxruntime_gpu-${ORT_VER}.dist-info && \
    mkdir -p "$STUB" && \
    printf "Metadata-Version: 2.1\nName: onnxruntime-gpu\nVersion: %s\n" "$ORT_VER" > "$STUB/METADATA" && \
    echo "Wheel-Version: 1.0\nGenerator: lab10-stub\nRoot-Is-Purelib: true" > "$STUB/WHEEL" && \
    : > "$STUB/RECORD"

# 6. COPY CODE LAST
# Because this is at the bottom, changing a Python file will NO LONGER trigger pip installs!
COPY src/ /app/src/
RUN mkdir -p /app/models

ENV PYTHONPATH=/app/src

# 7. Command to start your driver monitoring system
CMD ["python3", "src/dms/main.py", "--no-window"]
# ==========================================================
# STAGE 1: The Builder (Heavy tools, compilers, and PyTorch)
# ==========================================================
FROM dustynv/pytorch:2.7-r36.4.0 AS builder

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && apt-get clean && rm -rf /var/lib/apt/lists/*

# Override dustynv's private pip mirror to use public PyPI
ENV PIP_INDEX_URL=https://pypi.org/simple/
ENV PIP_EXTRA_INDEX_URL=

WORKDIR /build

# 1. Install our project's pip dependencies into a staging folder (/install)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

RUN apt-get update && apt-get install -y python3-pip && \
    rm -rf /install/local/lib/python3.10/dist-packages/onnxruntime* && \
    pip3 install --no-cache-dir --prefix=/install https://github.com/ultralytics/assets/releases/download/v0.0.0/onnxruntime_gpu-1.23.0-cp310-cp310-linux_aarch64.whl

# Tell Python where to find the staged packages so the rest of the build works
ENV PATH=/install/local/bin:$PATH
ENV PYTHONPATH=/install/local/lib/python3.10/dist-packages

# 2. Stub ONNX so Ultralytics doesn't crash on startup
RUN ORT_VER=$(python3 -c "import onnxruntime; print(onnxruntime.__version__)") && \
    STUB=/install/local/lib/python3.10/dist-packages/onnxruntime_gpu-${ORT_VER}.dist-info && \
    mkdir -p "$STUB" && \
    printf "Metadata-Version: 2.1\nName: onnxruntime-gpu\nVersion: %s\n" "$ORT_VER" > "$STUB/METADATA" && \
    echo "Wheel-Version: 1.0\nGenerator: lab10-stub\nRoot-Is-Purelib: true" > "$STUB/WHEEL" && \
    : > "$STUB/RECORD"

# 3. Copy your entire models directory and compile YOLO26 into a TensorRT engine
COPY models/ /build/models/
RUN cd /build/models && \
    python3 -c "from ultralytics import YOLO; YOLO('yolo26n.pt', task='detect').export(format='engine', imgsz=320, half=True, opset=19)"

# 4. Stage the native CUDA and cuDNN shared libraries (.so files)
RUN mkdir -p /staging_libs/cuda /staging_libs/aarch64 && \
    cd /usr/local/cuda/lib64 && cp -a *.so* /staging_libs/cuda/ && \
    cd /usr/lib/aarch64-linux-gnu && cp -a libcudnn*.so* libnvinfer*.so* libnvonnx*.so* /staging_libs/aarch64/


# ==========================================================
# STAGE 2: The Runtime (Lightweight, lean, production-ready)
# ==========================================================
FROM nvcr.io/nvidia/l4t-base:r36.2.0

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install basic OS-level requirements (bypassing invalid GPG signatures)
RUN apt-get update --allow-insecure-repositories || true
RUN apt-get install -y --allow-unauthenticated --no-install-recommends \
    python3 python3-pip libgl1-mesa-glx libglib2.0-0 alsa-utils libgomp1 libopenblas0 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 2. Copy all the pip packages we installed in Stage 1
COPY --from=builder /install/local/ /usr/local/

# 3. Copy the base PyTorch packages from the NVIDIA dustynv image
COPY --from=builder /usr/local/lib/python3.10/dist-packages/ /usr/local/lib/python3.10/dist-packages/

# 4. Copy the CUDA/cuDNN hardware libraries and register them with the system
RUN mkdir -p /usr/local/cuda/lib64
COPY --from=builder /staging_libs/cuda/ /usr/local/cuda/lib64/
COPY --from=builder /staging_libs/aarch64/ /usr/lib/aarch64-linux-gnu/
RUN echo '/usr/local/cuda/lib64' > /etc/ld.so.conf.d/cuda.conf && ldconfig

# 5. Copy YOUR actual project code and ALL models (ONNX + the new YOLO engine)
WORKDIR /app
COPY src/ /app/src/
RUN mkdir -p /app/models
# Copy the ONNX models and the compiled TensorRT engine from the builder
COPY --from=builder /build/models/*.onnx /app/models/
COPY --from=builder /build/models/yolo26n.engine /app/models/

# Tell Python where to find the 'dms' module
ENV PYTHONPATH=/app/src

# 6. Command to start your driver monitoring system
CMD ["python3", "src/dms/main.py", "--no-window"]
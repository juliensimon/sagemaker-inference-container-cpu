# Multi-stage build to separate build dependencies from runtime
FROM arm64v8/ubuntu:24.04 AS builder

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    build-essential cmake pkg-config \
    python3 python3-pip python3-venv \
    libcurl4-openssl-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set up build directories
ENV LLAMACPP_DIR=/opt/llama.cpp \
    VENV_DIR=/opt/venv

# Clone and build llama.cpp
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ${LLAMACPP_DIR}
WORKDIR ${LLAMACPP_DIR}
RUN cmake -S . -B build \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_BENCHMARKS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DLLAMA_BLAS=ON \
    -DLLAMA_OPENBLAS=ON \
    -DLLAMA_ARM_NEON=ON \
    -DCMAKE_CXX_FLAGS="-march=native -mtune=native" \
    && cmake --build build --target llama-server llama-quantize -j"$(nproc)"

# Create Python venv and install requirements
RUN python3 -m venv ${VENV_DIR}
ENV PATH="${VENV_DIR}/bin:${PATH}"

# Copy requirements and install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir -r /tmp/requirements.txt && \
    pip install --no-cache-dir -r ${LLAMACPP_DIR}/requirements.txt

# Final runtime image
FROM arm64v8/ubuntu:24.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl \
    python3 \
    libcurl4-openssl-dev \
    libopenblas0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set up runtime directories and environment
ENV APP_DIR=/opt/app \
    MODELS_DIR=/opt/models \
    LLAMACPP_DIR=/opt/llama.cpp \
    VENV_DIR=/opt/venv \
    PORT=8080 \
    UPSTREAM_PORT=8081 \
    PATH="/opt/venv/bin:${PATH}"

RUN mkdir -p ${APP_DIR} ${MODELS_DIR} ${LLAMACPP_DIR} /root/.cache/huggingface

# Copy built binaries from builder stage
COPY --from=builder /opt/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize
COPY --from=builder /opt/llama.cpp/convert_hf_to_gguf.py ${LLAMACPP_DIR}/convert_hf_to_gguf.py
COPY --from=builder /opt/llama.cpp/gguf-py ${LLAMACPP_DIR}/gguf-py

# Copy Python virtual environment
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY app ${APP_DIR}/app
COPY start.sh ${APP_DIR}/start.sh
RUN chmod +x ${APP_DIR}/start.sh

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -sf http://127.0.0.1:${PORT}/ping || exit 1

ENTRYPOINT ["/opt/app/start.sh"]
CMD ["serve"]

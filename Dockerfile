FROM arm64v8/ubuntu:22.04

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install all dependencies in one layer
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git \
    build-essential cmake pkg-config \
    python3 python3-pip python3-venv \
    libcurl4-openssl-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set up directories
ENV APP_DIR=/opt/app \
    MODELS_DIR=/opt/models \
    LLAMACPP_DIR=/opt/llama.cpp \
    VENV_DIR=/opt/venv \
    PORT=8080 \
    UPSTREAM_PORT=8081

RUN mkdir -p ${APP_DIR} ${MODELS_DIR} ${LLAMACPP_DIR}

# Clone and build llama.cpp
RUN git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ${LLAMACPP_DIR}
WORKDIR ${LLAMACPP_DIR}
RUN cmake -S . -B build -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_BENCHMARKS=OFF -DBUILD_SHARED_LIBS=OFF && \
    cmake --build build --target llama-server llama-quantize -j && \
    ln -sf ${LLAMACPP_DIR}/build/bin/llama-server /usr/local/bin/llama-server && \
    ln -sf ${LLAMACPP_DIR}/build/bin/llama-quantize /usr/local/bin/llama-quantize

# Create Python venv and install requirements
RUN python3 -m venv ${VENV_DIR}
ENV PATH="${VENV_DIR}/bin:${PATH}"

# Copy and install app requirements
COPY requirements.txt ${APP_DIR}/requirements.txt
RUN pip install --no-cache-dir --upgrade pip==24.0 && \
    pip install --no-cache-dir -r ${APP_DIR}/requirements.txt

# Install llama.cpp conversion script dependencies in the same venv
WORKDIR ${LLAMACPP_DIR}
RUN /opt/venv/bin/pip install --no-cache-dir -r ${LLAMACPP_DIR}/requirements.txt

# Copy application code
COPY app ${APP_DIR}/app
COPY start.sh ${APP_DIR}/start.sh
RUN chmod +x ${APP_DIR}/start.sh

# Create HF cache directory
RUN mkdir -p /root/.cache/huggingface

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -sf http://127.0.0.1:${PORT}/ping || exit 1

ENTRYPOINT ["/opt/app/start.sh"]
CMD ["serve"]

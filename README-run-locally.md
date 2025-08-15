# Running the Container Locally with Docker

This guide shows you how to test the SageMaker inference container locally using Docker, without needing Amazon SageMaker or the SageMaker SDK.

## Prerequisites

- Docker with ARM64/multi-platform support
- HuggingFace account and token (for gated models)
- Sufficient disk space (models can be several GB)

## Quick Start

### 1. Build the Container

```bash
docker build --platform linux/arm64 -t sagemaker-inference-llamacpp-graviton .
```

### 2. Run with a Public Model

For public models that don't require authentication:

```bash
docker run -d -p 8080:8080 \
  -e HF_MODEL_ID="microsoft/DialoGPT-medium" \
  -e QUANTIZATION="Q4_K_M" \
  --name llm-test \
  sagemaker-inference-llamacpp-graviton
```

### 3. Run with a Private or Gated Model

For private and gated models and persistent storage:

#### First Run (Download + Convert + Quantize)
```bash
# Create local directory for models
mkdir -p ./local_models

# First run: Download, convert, and quantize the model
docker run -d -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/local_models:/opt/models \
  -e HF_MODEL_ID="arcee-ai/AFM-4.5B" \
  -e HF_TOKEN="your_hf_token_here" \
  -e QUANTIZATION="Q4_K_M" \
  --name llm-test \
  sagemaker-inference-llamacpp-graviton
```

#### Subsequent Runs (Use Existing Quantized Model)
```bash
# Subsequent runs: Use the existing quantized model (much faster!)
docker run -d -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/local_models:/opt/models \
  -e MODEL_FILENAME="model-f16.Q4_K_M.gguf" \
  --name llm-test \
  sagemaker-inference-llamacpp-graviton
```

**Important:**
- Replace `your_hf_token_here` with your actual HuggingFace token from [HuggingFace tokens page](https://huggingface.co/settings/tokens) or `~/.cache/huggingface/token`
- **First run**: Takes 5+ minutes (download, convert, quantize)
- **Subsequent runs**: Takes ~30 seconds (loads existing model directly)

## Environment Variables

| Variable | Description | Required | Example |
|----------|-------------|----------|---------|
| `HF_MODEL_ID` | HuggingFace model repository | Yes | `arcee-ai/AFM-4.5B` |
| `HF_TOKEN` | HuggingFace token for private/gated models | For gated models | `hf_xxxxxxxxxxxx` |
| `QUANTIZATION` | Quantization level | No (default: F16) | `Q4_K_M`, `Q8_0` |
| `LLAMA_CPP_ARGS` | Additional llama-server arguments | No | `"--ctx-size 4096"` |
| `MODEL_FILENAME` | Specific GGUF file (for pre-quantized models) | For GGUF models | `model.q4_k_m.gguf` |

## Volume Mounts Explained

### HuggingFace Cache Mount
```bash
-v ~/.cache/huggingface:/root/.cache/huggingface
```
- **Purpose:** Reuses downloaded models from your local HuggingFace cache
- **Benefit:** Avoids re-downloading models you already have
- **Contains:** Model files, tokens, and HuggingFace metadata

### Models Directory Mount
```bash
-v $(pwd)/local_models:/opt/models
```
- **Purpose:** Persists converted and quantized models locally
- **Benefit:** Subsequent runs skip conversion/quantization (much faster!)
- **Contains:** Original model files, GGUF conversions, and quantized models

## What Happens During Startup

### First Run (with HF_MODEL_ID + QUANTIZATION)
1. **Download:** Model is downloaded from HuggingFace Hub
2. **Convert:** Safetensors/PyTorch model → GGUF F16 format
3. **Quantize:** F16 model → Quantized model (Q4_K_M reduces size by ~70%)
4. **Start:** llama.cpp server starts with the quantized model
5. **Ready:** FastAPI adapter starts and proxies requests

### Subsequent Runs (with MODEL_FILENAME)
1. **Skip Download/Convert/Quantize:** Uses existing quantized model directly
2. **Start:** llama.cpp server starts with the existing quantized model
3. **Ready:** FastAPI adapter starts and proxies requests

**Performance Comparison:**
- **First run**: 5+ minutes (download + convert + quantize)
- **Subsequent runs**: ~30 seconds (direct model loading)

**Example file sizes for AFM-4.5B:**
- Original safetensors: 8.6GB
- GGUF F16: 8.6GB
- GGUF Q4_K_M: 2.7GB (68% reduction!)

## Testing the APIs

### Health Check
```bash
curl http://localhost:8080/ping
# Expected: OK
```

### OpenAI-Compatible Chat Completions
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Hello! How are you?"}
    ],
    "max_tokens": 100,
    "temperature": 0.7
  }'
```

### OpenAI-Compatible Completions
```bash
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "The benefits of small language models include:",
    "max_tokens": 80,
    "temperature": 0.7
  }'
```

### SageMaker-Style Invocations
```bash
curl -X POST http://localhost:8080/invocations \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Write a haiku about Docker"}
    ],
    "max_tokens": 50
  }'
```

### Streaming Responses
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count from 1 to 5"}
    ],
    "max_tokens": 30,
    "stream": true
  }'
```

## Model Configuration Examples

### Using a Pre-quantized GGUF Model
```bash
docker run -d -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/local_models:/opt/models \
  -e HF_MODEL_ID="TheBloke/Llama-2-7B-Chat-GGUF" \
  -e MODEL_FILENAME="llama-2-7b-chat.q4_k_m.gguf" \
  --name llm-test \
  sagemaker-inference-llamacpp-graviton
```

### Custom llama.cpp Arguments
```bash
docker run -d -p 8080:8080 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/local_models:/opt/models \
  -e HF_MODEL_ID="microsoft/DialoGPT-medium" \
  -e QUANTIZATION="Q4_K_M" \
  -e LLAMA_CPP_ARGS="--ctx-size 4096 --threads 8" \
  --name llm-test \
  sagemaker-inference-llamacpp-graviton
```

## Finding Your MODEL_FILENAME

After the first run, you can find the quantized model filename by listing the files:

```bash
ls -la ./local_models/current/*.gguf
```

The filename pattern depends on your quantization setting:
- `Q4_K_M` → `model-f16.Q4_K_M.gguf`
- `Q8_0` → `model-f16.Q8_0.gguf`
- `F16` (no quantization) → `model-f16.gguf`

Use this filename in the `MODEL_FILENAME` environment variable for subsequent runs.

## File Structure After Running

```
local_models/
└── current/
    ├── config.json                      # Model configuration
    ├── model-00001-of-00002.safetensors # Original model (part 1)
    ├── model-00002-of-00002.safetensors # Original model (part 2)
    ├── model-f16.gguf                   # Converted F16 GGUF
    ├── model-f16.Q4_K_M.gguf           # Quantized model (used by server)
    ├── tokenizer.json                   # Tokenizer
    └── ... (other model files)
```

The quantized `.gguf` file is what the llama.cpp server actually uses for inference.

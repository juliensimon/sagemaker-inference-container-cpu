# Docker Compose Setup for AFM-4.5B

This guide shows how to run the AFM-4.5B model using Docker Compose with different configurations optimized for various use cases.

## Prerequisites

- Docker and Docker Compose installed
- ARM64/multi-platform support
- HuggingFace token for accessing AFM-4.5B (gated model)
- Sufficient disk space (~15GB for full model + conversions)

## Setup

### 1. Create Environment File

Create a `.env` file in the project root with your HuggingFace token:

```bash
# Create .env file
echo "HF_TOKEN=your_hf_token_here" > .env
```

Replace `your_hf_token_here` with your actual token from the [HuggingFace tokens page](https://huggingface.co/settings/tokens).

### 2. Create Models Directory

```bash
mkdir -p ./local_models
```

## Usage

The Docker Compose file includes several service profiles for different scenarios:

### First Run (Download + Convert + Quantize)

For the initial setup, run the first-run profile to download, convert, and quantize the model:

```bash
# Build and start first run (takes 5+ minutes)
docker-compose --profile first-run up --build afm-first-run
```

This will:
- Download AFM-4.5B from HuggingFace (requires HF_TOKEN)
- Convert from safetensors to GGUF F16 format
- Quantize to Q4_K_M (68% size reduction)
- Start the inference server

### Subsequent Runs (Fast Startup)

After the first run completes, use the fast profile for quick startups:

```bash
# Stop first run
docker-compose --profile first-run down

# Start fast run (takes ~30 seconds)
docker-compose --profile fast up afm-fast
```

This uses the existing quantized model (`model-f16.Q4_K_M.gguf`) for near-instant startup.

### Alternative Quantization Levels

#### Q8_0 Quantization (Higher Quality)
```bash
docker-compose --profile q8 up --build afm-q8
```

#### No Quantization (F16 - Highest Quality)
```bash
docker-compose --profile f16 up --build afm-f16
```

## Service Profiles

| Profile | Service | Purpose | Startup Time | Model Quality | Size |
|---------|---------|---------|--------------|---------------|------|
| `first-run` | `afm-first-run` | Initial setup with Q4_K_M | 5+ minutes | Good | 2.7GB |
| `fast` | `afm-fast` | Use existing Q4_K_M model | ~30 seconds | Good | 2.7GB |
| `q8` | `afm-q8` | Higher quality quantization | 5+ minutes | Very High | ~4.6GB |
| `f16` | `afm-f16` | No quantization (maximum quality) | 5+ minutes | Highest | 8.6GB |

## Testing the Service

### Health Check
```bash
curl http://localhost:8080/ping
```

### Chat Completions API
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are AFM, a helpful AI assistant."},
      {"role": "user", "content": "Hello! What can you tell me about yourself?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

### Streaming Response
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Count from 1 to 10"}
    ],
    "max_tokens": 50,
    "stream": true
  }'
```

## Service Management

### View Logs
```bash
# Follow logs for first-run
docker-compose --profile first-run logs -f afm-first-run

# Follow logs for fast startup
docker-compose --profile fast logs -f afm-fast
```

### Stop Services
```bash
# Stop specific profile
docker-compose --profile first-run down
docker-compose --profile fast down

# Stop all services
docker-compose down
```

### Check Service Status
```bash
# List running services
docker-compose ps

# Check health status
docker-compose --profile fast ps afm-fast
```

## Volume Mounts

The compose file uses two volume mounts:

### HuggingFace Cache
```yaml
- ~/.cache/huggingface:/root/.cache/huggingface
```
- Reuses your local HuggingFace cache
- Avoids re-downloading models
- Shares authentication tokens

### Models Directory
```yaml
- ./local_models:/opt/models
```
- Persists converted and quantized models
- Enables fast subsequent startups
- Allows model reuse across containers

## Environment Variables

| Variable | Description | Used In | Example |
|----------|-------------|---------|---------|
| `HF_TOKEN` | HuggingFace authentication token | first-run, q8, f16 | `hf_xxxxxxxxxxxx` |
| `HF_MODEL_ID` | Model repository ID | first-run, q8, f16 | `arcee-ai/AFM-4.5B` |
| `QUANTIZATION` | Quantization level | first-run, q8 | `Q4_K_M`, `Q8_0` |
| `MODEL_FILENAME` | Existing model file | fast | `model-f16.Q4_K_M.gguf` |

## Development Workflow

### 1. Initial Development
```bash
# First time setup
docker-compose --profile first-run up --build afm-first-run
```

### 2. Development Iterations
```bash
# Quick restarts during development
docker-compose --profile fast up afm-fast
```

### 3. Testing Different Quantizations
```bash
# Test Q8_0 quantization
docker-compose --profile q8 up --build afm-q8

# Test without quantization
docker-compose --profile f16 up --build afm-f16
```

### 4. Production Deployment
```bash
# Use fast profile for production
docker-compose --profile fast up -d afm-fast
```

## Troubleshooting

### Authentication Issues
```bash
# Verify HF_TOKEN is set correctly
cat .env

# Check token validity
docker-compose --profile first-run run afm-first-run sh -c "echo $HF_TOKEN"
```

### Storage Issues
```bash
# Check available disk space
df -h

# Check model files
ls -lah ./local_models/current/
```

### Service Health
```bash
# Check container logs
docker-compose --profile fast logs afm-fast

# Check service health
curl -f http://localhost:8080/ping || echo "Service not ready"
```

### Port Conflicts
If port 8080 is already in use, modify the docker-compose.yml:

```yaml
ports:
  - "8081:8080"  # Use port 8081 instead
```

## Performance Optimization

### Resource Limits
Add resource limits for production:

```yaml
deploy:
  resources:
    limits:
      memory: 16G
      cpus: '8'
    reservations:
      memory: 8G
      cpus: '4'
```

### Custom llama.cpp Arguments
Add custom arguments to any service:

```yaml
environment:
  - MODEL_FILENAME=model-f16.Q4_K_M.gguf
  - LLAMA_CPP_ARGS=--ctx-size 4096 --threads 8
```

## File Structure

After running the first-run profile, your directory structure will be:

```
.
├── docker-compose.yml
├── .env                              # HuggingFace token
├── local_models/
│   └── current/
│       ├── config.json
│       ├── model-00001-of-00002.safetensors
│       ├── model-00002-of-00002.safetensors
│       ├── model-f16.gguf            # F16 conversion
│       ├── model-f16.Q4_K_M.gguf     # Q4_K_M quantization
│       ├── model-f16.Q8_0.gguf       # Q8_0 quantization (if created)
│       └── tokenizer.json
└── ... (other project files)
```

This Docker Compose setup provides a flexible, production-ready way to deploy and manage the AFM-4.5B model with different configurations optimized for various use cases.

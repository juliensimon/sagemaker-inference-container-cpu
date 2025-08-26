# Multi-Architecture AFM-4.5B Inference Container

This repository provides a multi-architecture Docker container for running the AFM-4.5B model on both ARM64 and AMD64 platforms with architecture-specific optimizations.

## 🏗️ Repository Structure

```
sagemaker-inference-container-graviton/
├── docker/
│   ├── arm64/                    # ARM64-specific configurations
│   │   ├── Dockerfile           # ARM64-optimized Dockerfile
│   │   └── docker-compose.yml   # ARM64-specific compose file
│   ├── amd64/                    # AMD64/Intel-specific configurations
│   │   ├── Dockerfile           # AMD64-optimized Dockerfile
│   │   └── docker-compose.yml   # AMD64-specific compose file
│   └── multiarch/                # Multi-architecture configurations
│       ├── Dockerfile           # Multi-arch Dockerfile
│       └── docker-compose.yml   # Multi-arch compose file
├── scripts/
│   ├── build-multiarch.sh       # Build for all architectures
│   ├── build-arm64.sh           # Build for ARM64 only
│   ├── build-amd64.sh           # Build for AMD64 only
│   └── detect-architecture.sh   # Auto-detect and configure
├── config/
│   ├── arm64/                   # ARM64-specific build configs
│   ├── amd64/                   # AMD64-specific build configs
│   └── common/                  # Shared configurations
├── docs/
│   ├── arm64-setup.md          # ARM64 setup guide
│   ├── amd64-setup.md          # AMD64 setup guide
│   └── multiarch-deployment.md # Multi-arch deployment guide
├── app/                         # Shared application code
└── tests/                       # Architecture-specific tests
```

## 🚀 Quick Start

### 1. Auto-Detect Your Architecture

```bash
# This will automatically configure everything for your platform
source scripts/detect-architecture.sh
```

### 2. Build for Your Platform

```bash
# Build for your detected architecture
./scripts/build-$ARCH_NAME.sh

# Or build for all architectures
./scripts/build-multiarch.sh
```

### 3. Run the Service

```bash
# First run (download, convert, quantize)
docker-compose -f $COMPOSE_FILE --profile first-run up --build afm-first-run

# Subsequent runs (fast startup)
docker-compose -f $COMPOSE_FILE --profile fast up afm-fast
```

## 📋 Prerequisites

- Docker and Docker Compose installed
- HuggingFace token for AFM-4.5B (gated model)
- Sufficient disk space (~15GB for full model + conversions)

## 🔧 Build Options

### Single Architecture Build
```bash
# ARM64 only
./scripts/build-arm64.sh

# AMD64 only
./scripts/build-amd64.sh
```

## 🚀 Deployment Options

### Option 1: Auto-Detection (Recommended)
```bash
source scripts/detect-architecture.sh
docker-compose -f $COMPOSE_FILE --profile fast up afm-fast
```

### Option 2: Manual Selection
```bash
# ARM64
docker-compose -f docker/arm64/docker-compose.yml --profile fast up afm-fast

# AMD64
docker-compose -f docker/amd64/docker-compose.yml --profile fast up afm-fast

## 📊 Performance Comparison

| Metric | ARM64 | AMD64 | Notes |
|--------|-------|-------|-------|
| Build Time | ~15-20 min | ~10-15 min | AMD64 typically faster |
| Startup Time | ~30-45s | ~25-35s | Depends on hardware |
| Inference Speed | ~12-20 tokens/s | ~15-25 tokens/s | CPU-dependent |
| Memory Usage | ~8GB | ~8GB | Similar across platforms |
| Power Efficiency | Better | Good | ARM64 more efficient |

## 🧪 Testing

### Health Check
```bash
curl http://localhost:8080/ping
```

### API Test
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Hello, how are you?"}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## 📚 Documentation

- [ARM64 Setup Guide](docs/arm64-setup.md) - Detailed ARM64 setup and optimization
- [AMD64 Setup Guide](docs/amd64-setup.md) - Detailed AMD64 setup and optimization
- [Original Docker Compose Guide](README-docker-compose.md) - Original setup guide

## 🔍 Troubleshooting

### Common Issues

1. **Build failures**: Ensure you have the correct Docker platform support
2. **Performance issues**: Check thread count and memory allocation
3. **Model loading errors**: Verify sufficient disk space and memory

### Debug Commands

```bash
# Check architecture
uname -m

# Check Docker platform
docker version

# Check container logs
docker-compose -f $COMPOSE_FILE logs afm-fast

# Check resource usage
docker stats
```

## 🤝 Contributing

When contributing to this multi-architecture setup:

1. **Test on both platforms**: Ensure changes work on ARM64 and AMD64
2. **Update documentation**: Keep architecture-specific guides current
3. **Add tests**: Include tests for both architectures
4. **Performance testing**: Benchmark changes on both platforms

## 📄 License

This project is licensed under the same terms as the original repository.

## 🙏 Acknowledgments

- Original AFM-4.5B model by Arcee AI
- llama.cpp for the inference engine
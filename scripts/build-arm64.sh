#!/bin/bash

# Build specifically for ARM64 architecture
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building ARM64 Docker image...${NC}"

# Check if we're on an ARM64 system or have ARM64 emulation
CURRENT_ARCH=$(uname -m)
if [ "$CURRENT_ARCH" != "aarch64" ] && [ "$CURRENT_ARCH" != "arm64" ]; then
    echo -e "${YELLOW}Warning: Building ARM64 on non-ARM64 system ($CURRENT_ARCH)${NC}"
    echo -e "${YELLOW}This requires ARM64 emulation support.${NC}"
    echo -e "${YELLOW}If the build fails, you may need to:${NC}"
    echo -e "${YELLOW}1. Install QEMU: sudo apt-get install qemu-user-static${NC}"
    echo -e "${YELLOW}2. Enable Docker emulation: docker run --rm --privileged multiarch/qemu-user-static --reset -p yes${NC}"
    echo -e "${YELLOW}3. Or build on an actual ARM64 system${NC}"
    echo ""
fi

# Get the current directory (should be the project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Build the ARM64 image
docker build \
    --platform linux/arm64 \
    --file docker/arm64/Dockerfile \
    --tag afm-inference:arm64 \
    --tag afm-inference:latest \
    "$PROJECT_ROOT"

echo -e "${GREEN}ARM64 build completed successfully!${NC}"
echo -e "${YELLOW}Available tags:${NC}"
echo "  - afm-inference:arm64"
echo "  - afm-inference:latest"

# Optional: Push to registry (uncomment and modify as needed)
# echo -e "${YELLOW}To push to a registry, uncomment the push command in the script${NC}"
# docker tag afm-inference:arm64 your-registry/afm-inference:arm64
# docker push your-registry/afm-inference:arm64

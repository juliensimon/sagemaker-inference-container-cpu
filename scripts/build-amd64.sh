#!/bin/bash

# Build specifically for AMD64 architecture
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Building AMD64 Docker image...${NC}"

# Get the current directory (should be the project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Build the AMD64 image
docker build \
    --platform linux/amd64 \
    --file docker/amd64/Dockerfile \
    --tag afm-inference:amd64 \
    --tag afm-inference:latest \
    "$PROJECT_ROOT"

echo -e "${GREEN}AMD64 build completed successfully!${NC}"
echo -e "${YELLOW}Available tags:${NC}"
echo "  - afm-inference:amd64"
echo "  - afm-inference:latest"

# Optional: Push to registry (uncomment and modify as needed)
# echo -e "${YELLOW}To push to a registry, uncomment the push command in the script${NC}"
# docker tag afm-inference:amd64 your-registry/afm-inference:amd64
# docker push your-registry/afm-inference:amd64

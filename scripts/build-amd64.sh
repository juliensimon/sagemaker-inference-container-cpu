#!/bin/bash

# Build specifically for AMD64 architecture
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Parse command line arguments
FORCE_REBUILD=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --force-rebuild)
            FORCE_REBUILD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--force-rebuild]"
            echo "  --force-rebuild    Force a complete rebuild without using cache"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

echo -e "${GREEN}Building AMD64 Docker image...${NC}"
if [ "$FORCE_REBUILD" = true ]; then
    echo -e "${YELLOW}Force rebuild enabled - no cache will be used${NC}"
fi

# Get the current directory (should be the project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Build the AMD64 image
BUILD_ARGS=(
    --platform linux/amd64
    --file docker/amd64/Dockerfile
    --tag afm-inference:amd64
    --tag afm-inference:latest
)

# Add --no-cache flag if force rebuild is enabled
if [ "$FORCE_REBUILD" = true ]; then
    BUILD_ARGS+=(--no-cache)
fi

docker build "${BUILD_ARGS[@]}" "$PROJECT_ROOT"

echo -e "${GREEN}AMD64 build completed successfully!${NC}"
echo -e "${YELLOW}Available tags:${NC}"
echo "  - afm-inference:amd64"
echo "  - afm-inference:latest"

# Optional: Push to registry (uncomment and modify as needed)
# echo -e "${YELLOW}To push to a registry, uncomment the push command in the script${NC}"
# docker tag afm-inference:amd64 your-registry/afm-inference:amd64
# docker push your-registry/afm-inference:amd64

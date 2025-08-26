#!/bin/bash

# Detect current architecture and set appropriate configurations
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Detecting system architecture...${NC}"

# Detect architecture
ARCH=$(uname -m)
echo -e "${GREEN}Detected architecture: ${ARCH}${NC}"

# Set architecture-specific configurations
case $ARCH in
  x86_64)
    echo -e "${YELLOW}Setting up AMD64/Intel configuration...${NC}"
    export DOCKERFILE_PATH=docker/amd64/Dockerfile
    export COMPOSE_FILE=docker/amd64/docker-compose.yml
    export ARCH_NAME="amd64"
    export PLATFORM="linux/amd64"
    export BUILD_SCRIPT="scripts/build-amd64.sh"
    ;;
  aarch64|arm64)
    echo -e "${YELLOW}Setting up ARM64 configuration...${NC}"
    export DOCKERFILE_PATH=docker/arm64/Dockerfile
    export COMPOSE_FILE=docker/arm64/docker-compose.yml
    export ARCH_NAME="arm64"
    export PLATFORM="linux/arm64"
    export BUILD_SCRIPT="scripts/build-arm64.sh"
    ;;
  *)
    echo -e "${RED}Unsupported architecture: $ARCH${NC}"
    echo -e "${YELLOW}Available architectures:${NC}"
    echo "  - x86_64 (AMD64/Intel)"
    echo "  - aarch64/arm64 (ARM64)"
    exit 1
    ;;
esac

# Display configuration
echo -e "${GREEN}Architecture configuration:${NC}"
echo "  Architecture: $ARCH_NAME"
echo "  Platform: $PLATFORM"
echo "  Dockerfile: $DOCKERFILE_PATH"
echo "  Compose file: $COMPOSE_FILE"
echo "  Build script: $BUILD_SCRIPT"

# Export variables for use in other scripts
export ARCH
export ARCH_NAME
export PLATFORM
export DOCKERFILE_PATH
export COMPOSE_FILE
export BUILD_SCRIPT

echo -e "${GREEN}Architecture detection completed. Variables exported.${NC}"
echo -e "${YELLOW}To use these variables in another script, source this file:${NC}"
echo "  source scripts/detect-architecture.sh"

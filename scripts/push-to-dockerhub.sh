#!/bin/bash

# Push Docker images to Docker Hub
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
DOCKER_USERNAME="juliensimon"
IMAGE_NAME="sagemaker-inference-llamacpp-cpu"
VERSION=""
PUSH_LATEST=false
PUSH_MULTIARCH=false
DRY_RUN=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --username|-u)
            DOCKER_USERNAME="$2"
            shift 2
            ;;
        --image-name|-i)
            IMAGE_NAME="$2"
            shift 2
            ;;
        --version|-v)
            VERSION="$2"
            shift 2
            ;;
        --latest)
            PUSH_LATEST=true
            shift
            ;;
        --multiarch)
            PUSH_MULTIARCH=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Push Docker images to Docker Hub"
            echo ""
            echo "Options:"
            echo "  -u, --username USERNAME    Docker Hub username (default: juliensimon)"
            echo "  -i, --image-name NAME      Image name (default: sagemaker-inference-llamacpp-cpu)"
            echo "  -v, --version VERSION      Version tag to push (e.g., 1.0.0)"
            echo "  --latest                   Also push as 'latest' tag"
            echo "  --multiarch                Push multi-architecture manifest"
            echo "  --dry-run                  Show what would be pushed without actually pushing"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --version 1.0.0 --latest"
            echo "  $0 --multiarch --version 1.0.0"
            echo "  $0 --dry-run --version 1.0.0"
            echo ""
            echo "Prerequisites:"
            echo "  - Docker must be installed and running"
            echo "  - You must be logged in to Docker Hub: docker login"
            echo "  - Images must be built first using build-amd64.sh or build-arm64.sh"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate prerequisites
echo -e "${BLUE}Checking prerequisites...${NC}"

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running or not accessible${NC}"
    exit 1
fi

# Check if logged in to Docker Hub (simplified check)
echo -e "${GREEN}Assuming Docker Hub login is configured${NC}"

# Check if version is provided
if [ -z "$VERSION" ]; then
    echo -e "${RED}Error: Version is required${NC}"
    echo -e "${YELLOW}Use --version to specify a version tag${NC}"
    exit 1
fi

# Get the current directory (should be the project root)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Function to check if image exists locally
check_image_exists() {
    local image_tag="$1"
    docker image inspect "$image_tag" >/dev/null 2>&1
}

# Function to push image
push_image() {
    local local_tag="$1"
    local remote_tag="$2"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would push: $local_tag -> $remote_tag${NC}"
        return 0
    fi

    echo -e "${BLUE}Pushing $local_tag as $remote_tag...${NC}"
    docker tag "$local_tag" "$remote_tag"
    docker push "$remote_tag"
    echo -e "${GREEN}Successfully pushed $remote_tag${NC}"
}

# Function to create and push multi-architecture manifest
push_multiarch() {
    local version_tag="${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
    local latest_tag="${DOCKER_USERNAME}/${IMAGE_NAME}:latest"

    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}[DRY RUN] Would create multi-arch manifest:${NC}"
        echo -e "${YELLOW}  - $version_tag (amd64, arm64)${NC}"
        if [ "$PUSH_LATEST" = true ]; then
            echo -e "${YELLOW}  - $latest_tag (amd64, arm64)${NC}"
        fi
        return 0
    fi

    echo -e "${BLUE}Creating multi-architecture manifest...${NC}"

    # Create manifest for version tag
    docker manifest create "$version_tag" \
        "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-amd64" \
        "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-arm64"

    docker manifest push "$version_tag"
    echo -e "${GREEN}Successfully pushed multi-arch manifest: $version_tag${NC}"

    # Create manifest for latest tag if requested
    if [ "$PUSH_LATEST" = true ]; then
        docker manifest create "$latest_tag" \
            "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-amd64" \
            "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-arm64"

        docker manifest push "$latest_tag"
        echo -e "${GREEN}Successfully pushed multi-arch manifest: $latest_tag${NC}"
    fi
}

# Main push logic
echo -e "${GREEN}Starting Docker Hub push process...${NC}"
echo -e "${BLUE}Configuration:${NC}"
echo "  Username: $DOCKER_USERNAME"
echo "  Image name: $IMAGE_NAME"
echo "  Version: $VERSION"
echo "  Push latest: $PUSH_LATEST"
echo "  Multi-arch: $PUSH_MULTIARCH"
echo "  Dry run: $DRY_RUN"
echo ""

if [ "$PUSH_MULTIARCH" = true ]; then
    # Multi-architecture push
    echo -e "${BLUE}Multi-architecture push mode${NC}"

    # Check if both architecture images exist
    local_amd64_tag="afm-inference:amd64"
    local_arm64_tag="afm-inference:arm64"

    if ! check_image_exists "$local_amd64_tag"; then
        echo -e "${RED}Error: AMD64 image not found locally: $local_amd64_tag${NC}"
        echo -e "${YELLOW}Please build the AMD64 image first: ./scripts/build-amd64.sh${NC}"
        exit 1
    fi

    if ! check_image_exists "$local_arm64_tag"; then
        echo -e "${RED}Error: ARM64 image not found locally: $local_arm64_tag${NC}"
        echo -e "${YELLOW}Please build the ARM64 image first: ./scripts/build-arm64.sh${NC}"
        exit 1
    fi

    # Push individual architecture images
    push_image "$local_amd64_tag" "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-amd64"
    push_image "$local_arm64_tag" "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}-arm64"

    # Create and push multi-architecture manifest
    push_multiarch

else
    # Single architecture push
    echo -e "${BLUE}Single architecture push mode${NC}"

    # Detect current architecture
    CURRENT_ARCH=$(uname -m)
    case $CURRENT_ARCH in
        x86_64)
            ARCH_NAME="amd64"
            local_tag="afm-inference:amd64"
            ;;
        aarch64|arm64)
            ARCH_NAME="arm64"
            local_tag="afm-inference:arm64"
            ;;
        *)
            echo -e "${RED}Error: Unsupported architecture: $CURRENT_ARCH${NC}"
            exit 1
            ;;
    esac

    echo -e "${GREEN}Detected architecture: $ARCH_NAME${NC}"

    # Check if image exists locally
    if ! check_image_exists "$local_tag"; then
        echo -e "${RED}Error: Image not found locally: $local_tag${NC}"
        echo -e "${YELLOW}Please build the image first: ./scripts/build-${ARCH_NAME}.sh${NC}"
        exit 1
    fi

    # Push versioned image
    push_image "$local_tag" "${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"

    # Push latest if requested
    if [ "$PUSH_LATEST" = true ]; then
        push_image "$local_tag" "${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
    fi
fi

echo ""
echo -e "${GREEN}Push process completed successfully!${NC}"
echo -e "${BLUE}Pushed images:${NC}"

if [ "$PUSH_MULTIARCH" = true ]; then
    echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION} (multi-arch)"
    if [ "$PUSH_LATEST" = true ]; then
        echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:latest (multi-arch)"
    fi
else
    echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
    if [ "$PUSH_LATEST" = true ]; then
        echo "  - ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
    fi
fi

echo ""
echo -e "${YELLOW}You can now pull the image with:${NC}"
echo "  docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:${VERSION}"
if [ "$PUSH_LATEST" = true ]; then
    echo "  docker pull ${DOCKER_USERNAME}/${IMAGE_NAME}:latest"
fi

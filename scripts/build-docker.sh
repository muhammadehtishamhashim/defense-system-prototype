#!/bin/bash

# Docker build script for HifazatAI
# This script handles the multi-stage Docker build with proper error handling

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
IMAGE_NAME="hifazat-ai"
TAG="${1:-latest}"
DOCKERFILE="${2:-Dockerfile}"

echo -e "${GREEN}Building HifazatAI Docker image...${NC}"
echo "Image: ${IMAGE_NAME}:${TAG}"
echo "Dockerfile: ${DOCKERFILE}"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    exit 1
fi

# Check if required files exist
if [ ! -f "${DOCKERFILE}" ]; then
    echo -e "${RED}Error: Dockerfile '${DOCKERFILE}' not found${NC}"
    exit 1
fi

if [ ! -f "backend/requirements.prod.txt" ]; then
    echo -e "${RED}Error: backend/requirements.prod.txt not found${NC}"
    exit 1
fi

if [ ! -f "frontend/package.json" ]; then
    echo -e "${RED}Error: frontend/package.json not found${NC}"
    exit 1
fi

# Clean up any existing build artifacts
echo -e "${YELLOW}Cleaning up build artifacts...${NC}"
docker system prune -f --filter "label=stage=builder" > /dev/null 2>&1 || true

# Build the image with progress output
echo -e "${GREEN}Starting Docker build...${NC}"
echo "This may take several minutes for the first build..."
echo ""

# Build with detailed progress
docker build \
    --progress=plain \
    --tag "${IMAGE_NAME}:${TAG}" \
    --file "${DOCKERFILE}" \
    . 2>&1 | tee build.log

# Check if build was successful
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✓ Docker build completed successfully!${NC}"
    echo "Image: ${IMAGE_NAME}:${TAG}"
    
    # Show image size
    IMAGE_SIZE=$(docker images "${IMAGE_NAME}:${TAG}" --format "table {{.Size}}" | tail -n 1)
    echo "Size: ${IMAGE_SIZE}"
    
    # Test the image
    echo ""
    echo -e "${YELLOW}Testing the built image...${NC}"
    
    # Run a quick test to ensure the image works
    if docker run --rm "${IMAGE_NAME}:${TAG}" python -c "import backend.api.main; print('✓ Image test passed')"; then
        echo -e "${GREEN}✓ Image test passed${NC}"
    else
        echo -e "${YELLOW}⚠ Image test failed, but build completed${NC}"
    fi
    
else
    echo ""
    echo -e "${RED}✗ Docker build failed${NC}"
    echo "Check build.log for details"
    exit 1
fi

# Cleanup intermediate images
echo ""
echo -e "${YELLOW}Cleaning up intermediate images...${NC}"
docker image prune -f --filter "dangling=true" > /dev/null 2>&1 || true

echo ""
echo -e "${GREEN}Build process completed!${NC}"
echo ""
echo "To run the container:"
echo "  docker run -p 8000:8000 ${IMAGE_NAME}:${TAG}"
echo ""
echo "To run with docker-compose:"
echo "  docker-compose up"
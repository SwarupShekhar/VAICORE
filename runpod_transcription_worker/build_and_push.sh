#!/bin/bash
# Make sure you are logged into docker hub: `docker login`
# Usage: ./build_and_push.sh your_dockerhub_username

if [ -z "$1" ]
  then
    echo "Please provide your DockerHub username. Example: ./build_and_push.sh myusername"
    exit 1
fi

USERNAME=$1
IMAGE_NAME="stable-whisper-runpod"
TAG="latest"

echo "Building Docker image ${USERNAME}/${IMAGE_NAME}:${TAG}..."
docker build --platform linux/amd64 -t ${USERNAME}/${IMAGE_NAME}:${TAG} .

echo "Pushing image to DockerHub..."
docker push ${USERNAME}/${IMAGE_NAME}:${TAG}

echo "Done! You can now deploy ${USERNAME}/${IMAGE_NAME}:${TAG} on Runpod."

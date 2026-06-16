FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Install Python and FFmpeg
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip ffmpeg git wget \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
# Need to upgrade pip first
RUN pip3 install --no-cache-dir --upgrade pip
RUN pip3 install --no-cache-dir -r requirements.txt

# Pre-download the Whisper Large-v3 model to bake it into the image (prevents cold start delays)
RUN python3 -c "import whisper; whisper.load_model('large-v3')"

# Copy the handler
COPY handler.py .

# Run the handler
CMD ["python3", "-u", "handler.py"]

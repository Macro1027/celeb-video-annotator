# Use Python 3.12 slim image for smaller size
FROM python:3.12-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Redis server
    redis-server \
    # OpenCV dependencies
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Build tools for compiling Python packages
    build-essential \
    gcc \
    g++ \
    # Git for package installations
    git \
    # FFmpeg for video processing
    ffmpeg \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Create necessary directories
RUN mkdir -p temp results data config

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Install PyTorch first (large dependency)
RUN pip install --no-cache-dir torch>=1.13.0 torchvision>=0.14.0 --index-url https://download.pytorch.org/whl/cpu

# Install other requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy setup.py and install package
COPY setup.py .
COPY celeb_video_annotator/ ./celeb_video_annotator/

# Install the package in development mode
RUN pip install -e .

# Copy the rest of the application
COPY . .

# Make startup script executable
RUN chmod +x start_server.sh

# Create a non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose the port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set default environment variables
ENV HOST=0.0.0.0 \
    PORT=8000 \
    OUTPUT_DIR=results/ \
    DEBUG_CONFIG=0

# Run the startup script
CMD ["./start_server.sh"] 
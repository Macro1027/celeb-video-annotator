#!/bin/bash
set -e

echo "🚀 Starting Celebrity Video Annotator API..."

# Function to check if Redis is running
check_redis() {
    redis-cli ping >/dev/null 2>&1
}

# Start Redis if not running
if ! check_redis; then
    echo "📡 Starting Redis server..."
    redis-server --daemonize yes --port 6379
    
    # Wait for Redis to start
    for i in {1..10}; do
        if check_redis; then
            echo "✅ Redis started successfully"
            break
        fi
        echo "⏳ Waiting for Redis to start... (attempt $i/10)"
        sleep 2
    done
    
    if ! check_redis; then
        echo "❌ Failed to start Redis after 20 seconds"
        exit 1
    fi
else
    echo "✅ Redis is already running"
fi

# Create required directories
echo "📁 Creating required directories..."
mkdir -p temp results data config

# Check if config file exists, create basic one if not
if [ ! -f "config/config.yaml" ]; then
    echo "⚙️ Creating basic config file..."
    cat > config/config.yaml << EOF
# Celebrity Video Annotator Configuration
batch_size: 48
output_dir: results/
video_path: demo/sample.mp4
index_name: face-recognition-embeddings
target_label: null

# API key will be read from environment variable PINECONE_API_KEY
# Set it with: export PINECONE_API_KEY=your_key_here
EOF
    echo "✅ Basic config created at config/config.yaml"
fi

# Check for required environment variables
if [ -z "$PINECONE_API_KEY" ]; then
    echo "⚠️  Warning: PINECONE_API_KEY environment variable not set"
    echo "   Set it with: export PINECONE_API_KEY=your_key_here"
fi

# Set default environment variables if not set
export OUTPUT_DIR=${OUTPUT_DIR:-"results/"}
export DEBUG_CONFIG=${DEBUG_CONFIG:-"0"}

# Install package in development mode if not already installed
echo "📦 Ensuring package is installed..."
if ! python -c "import celeb_video_annotator" >/dev/null 2>&1; then
    echo "Installing package in development mode..."
    pip install -e .
fi

# Check if all dependencies are available
echo "🔍 Checking dependencies..."
python -c "
import sys
try:
    from celeb_video_annotator.api.endpoints import app
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    print('Run: pip install -r requirements.txt')
    sys.exit(1)
"

# Get the host and port from environment or use defaults
HOST=${HOST:-"0.0.0.0"}
PORT=${PORT:-8000}

echo "🌐 Starting FastAPI server..."
echo "   Host: $HOST"
echo "   Port: $PORT"
echo "   API Docs: http://localhost:$PORT/docs"
echo "   Health Check: http://localhost:$PORT/health"
echo ""
echo "🎬 Ready to process celebrity videos!"
echo "   Upload videos to: http://localhost:$PORT/annotate"
echo ""

# Start the FastAPI server
exec uvicorn celeb_video_annotator.api.endpoints:app \
    --host "$HOST" \
    --port "$PORT" \
    --reload \
    --log-level info 
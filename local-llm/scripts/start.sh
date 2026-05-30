#!/usr/bin/env bash
# Start the Local LLM API server
# Usage: bash start.sh [port] [gpu_layers]

PORT="${1:-8000}"
GPU_LAYERS="${2:--1}"   # -1 = all layers on GPU; 0 = CPU only

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
API_DIR="$SCRIPT_DIR/../api"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  🦙 Local LLM API Server"
echo "  Port       : $PORT"
echo "  GPU layers : $GPU_LAYERS"
echo "  API docs   : http://localhost:$PORT/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Install deps if needed
if ! python3 -c "import llama_cpp" &>/dev/null; then
    echo ""
    echo "Installing dependencies..."

    # Detect GPU
    if command -v nvidia-smi &>/dev/null; then
        echo "NVIDIA GPU detected — installing with CUDA support"
        CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python --upgrade
    elif [[ "$(uname)" == "Darwin" ]]; then
        echo "Apple Silicon detected — installing with Metal support"
        CMAKE_ARGS="-DLLAMA_METAL=on" pip install llama-cpp-python --upgrade
    else
        echo "No GPU detected — installing CPU-only"
        pip install llama-cpp-python --upgrade
    fi

    pip install -r "$API_DIR/requirements.txt"
fi

echo ""
export MODELS_DIR="$SCRIPT_DIR/../models"
export N_GPU_LAYERS="$GPU_LAYERS"

cd "$API_DIR"
exec uvicorn server:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --workers 1 \
    --log-level info

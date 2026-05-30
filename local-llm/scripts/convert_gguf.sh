#!/usr/bin/env bash
# Convert a HuggingFace model directory to GGUF format (for use with llama.cpp)
# Usage: bash convert_gguf.sh <model_dir> [quant_type]
#
# Requires llama.cpp to be cloned alongside this project:
#   git clone https://github.com/ggerganov/llama.cpp ../llama.cpp

set -e

MODEL_DIR="${1:?Usage: bash convert_gguf.sh <model_dir> [quant_type]}"
QUANT="${2:-q4_k_m}"
LLAMA_CPP="${LLAMA_CPP_DIR:-$(dirname "$0")/../../llama.cpp}"
MODELS_DIR="$(dirname "$0")/../models"
MODEL_NAME="$(basename "$MODEL_DIR")"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Converting: $MODEL_DIR"
echo "  Quant type: $QUANT"
echo "  llama.cpp:  $LLAMA_CPP"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -d "$LLAMA_CPP" ]; then
    echo "llama.cpp not found at $LLAMA_CPP"
    echo "Run: git clone https://github.com/ggerganov/llama.cpp"
    exit 1
fi

# Install conversion deps
pip install -q gguf sentencepiece transformers

# Step 1: Convert to unquantized GGUF (F16)
F16_PATH="$MODELS_DIR/${MODEL_NAME}.f16.gguf"
echo ""
echo "Step 1/2: Converting to F16 GGUF..."
python3 "$LLAMA_CPP/convert_hf_to_gguf.py" \
    "$MODEL_DIR" \
    --outfile "$F16_PATH" \
    --outtype f16

# Step 2: Quantize
QUANT_PATH="$MODELS_DIR/${MODEL_NAME}.${QUANT}.gguf"
echo ""
echo "Step 2/2: Quantizing to $QUANT..."

# Build quantize tool if needed
if [ ! -f "$LLAMA_CPP/llama-quantize" ]; then
    echo "Building llama-quantize..."
    make -C "$LLAMA_CPP" llama-quantize -j$(nproc)
fi

"$LLAMA_CPP/llama-quantize" "$F16_PATH" "$QUANT_PATH" "${QUANT^^}"

echo ""
echo "✅ Done!"
echo "   GGUF model: $QUANT_PATH"
echo "   (You can delete $F16_PATH to save space)"

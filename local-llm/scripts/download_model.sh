#!/usr/bin/env bash
# Download a GGUF model from HuggingFace
# Usage: bash download_model.sh [repo] [filename]

set -e

MODELS_DIR="$(dirname "$0")/../models"
mkdir -p "$MODELS_DIR"

# Defaults — Mistral 7B Q4_K_M (good balance of quality + speed)
REPO="${1:-TheBloke/Mistral-7B-Instruct-v0.2-GGUF}"
FILE="${2:-mistral-7b-instruct-v0.2.Q4_K_M.gguf}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Downloading: $FILE"
echo "  From:        $REPO"
echo "  Into:        $MODELS_DIR"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Try huggingface-cli first
if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$REPO" "$FILE" --local-dir "$MODELS_DIR"
else
    pip install -q huggingface_hub
    python3 -c "
from huggingface_hub import hf_hub_download
path = hf_hub_download(repo_id='$REPO', filename='$FILE', local_dir='$MODELS_DIR')
print(f'Saved to: {path}')
"
fi

echo ""
echo "✅ Done! Model saved to: $MODELS_DIR/$FILE"
echo ""
echo "Other popular models to try:"
echo "  7B  (fast):  bash download_model.sh TheBloke/Mistral-7B-Instruct-v0.2-GGUF mistral-7b-instruct-v0.2.Q4_K_M.gguf"
echo "  8B  (great): bash download_model.sh bartowski/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
echo "  13B (smart): bash download_model.sh TheBloke/Llama-2-13B-chat-GGUF llama-2-13b-chat.Q4_K_M.gguf"

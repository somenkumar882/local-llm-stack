# 🦙 Local LLM Stack

A complete, production-ready local LLM setup with GPU optimization, OpenAI-compatible API, and fine-tuning support.

## Project Structure

```
local-llm/
├── models/              # Store GGUF model files here
├── data/                # Training data (JSONL format)
├── checkpoints/         # Fine-tuning checkpoints
├── api/
│   ├── server.py        # FastAPI OpenAI-compatible server
│   └── requirements.txt
├── finetune/
│   ├── train.py         # LoRA/QLoRA fine-tuning
│   ├── merge.py         # Merge LoRA → base model
│   └── requirements.txt
├── scripts/
│   ├── download_model.sh  # Download GGUF from HuggingFace
│   ├── convert_gguf.sh    # Convert HF model → GGUF
│   └── start.sh           # Start the API server
└── docker-compose.yml   # Full stack with Open WebUI
```

## Quick Start

### 1. Download a Model
```bash
bash scripts/download_model.sh
```

### 2. Start the API Server
```bash
bash scripts/start.sh
```

### 3. Test It
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistral-7b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### 4. Open WebUI (Browser Chat)
```bash
docker-compose up -d
# Visit http://localhost:3000
```

## Fine-Tuning

```bash
# Prepare your data in data/train.jsonl
# Then run:
cd finetune
pip install -r requirements.txt
python train.py
python merge.py
bash ../scripts/convert_gguf.sh
```

## Hardware Guide

| Model  | Min RAM | GPU VRAM | Recommended Quant |
|--------|---------|----------|-------------------|
| 7B     | 8 GB    | 6 GB     | Q4_K_M            |
| 13B    | 16 GB   | 10 GB    | Q4_K_M            |
| 34B    | 32 GB   | 24 GB    | Q5_K_M            |
| 70B    | 64 GB   | 48 GB+   | Q4_K_M            |

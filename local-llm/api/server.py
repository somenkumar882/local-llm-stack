"""
Local LLM API Server — OpenAI-compatible
Run: uvicorn server:app --host 0.0.0.0 --port 8000 --reload
"""

import os
import json
import time
import uuid
import glob
from typing import List, Optional, Union
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Config ────────────────────────────────────────────────────────────────────
MODELS_DIR   = os.environ.get("MODELS_DIR", "../models")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")   # auto-detect if empty
N_GPU_LAYERS  = int(os.environ.get("N_GPU_LAYERS", "-1"))   # -1 = all on GPU
N_CTX         = int(os.environ.get("N_CTX", "4096"))
N_BATCH       = int(os.environ.get("N_BATCH", "512"))
N_THREADS     = int(os.environ.get("N_THREADS", "4"))

# ── Model Registry ────────────────────────────────────────────────────────────
loaded_models: dict = {}   # model_name → Llama instance

def find_gguf_files() -> dict[str, str]:
    """Returns {model_name: path} for all .gguf files in MODELS_DIR."""
    files = glob.glob(os.path.join(MODELS_DIR, "**/*.gguf"), recursive=True)
    return {os.path.splitext(os.path.basename(f))[0]: f for f in files}

def get_or_load_model(model_name: str):
    """Lazily load a model; return cached if already loaded."""
    if model_name in loaded_models:
        return loaded_models[model_name]

    registry = find_gguf_files()

    # Allow partial name match (e.g. "mistral" matches "mistral-7b-instruct-v0.2.Q4_K_M")
    matched = next(
        (path for name, path in registry.items() if model_name.lower() in name.lower()),
        None
    )
    if not matched:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found. Available: {list(registry.keys())}"
        )

    try:
        from llama_cpp import Llama
        print(f"[server] Loading model: {matched}")
        llm = Llama(
            model_path=matched,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            n_batch=N_BATCH,
            n_threads=N_THREADS,
            use_mmap=True,
            verbose=False,
        )
        loaded_models[model_name] = llm
        print(f"[server] Model '{model_name}' ready ✓")
        return llm
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}")

# ── Lifespan ──────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Pre-load default model on startup
    registry = find_gguf_files()
    target = DEFAULT_MODEL or (next(iter(registry), None))
    if target:
        try:
            get_or_load_model(target)
        except Exception as e:
            print(f"[server] Warning: could not pre-load model — {e}")
    else:
        print("[server] No .gguf files found in models/. Add one and restart.")
    yield
    loaded_models.clear()

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(title="Local LLM API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic Models ───────────────────────────────────────────────────────────
class Message(BaseModel):
    role: str       # "system" | "user" | "assistant"
    content: str

class ChatRequest(BaseModel):
    model: str                        = "auto"
    messages: List[Message]
    max_tokens: Optional[int]         = Field(512, ge=1, le=8192)
    temperature: Optional[float]      = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float]            = Field(0.95, ge=0.0, le=1.0)
    top_k: Optional[int]              = Field(40, ge=0)
    repeat_penalty: Optional[float]   = Field(1.1, ge=1.0)
    stream: Optional[bool]            = False
    stop: Optional[List[str]]         = None

class CompletionRequest(BaseModel):
    model: str                  = "auto"
    prompt: str
    max_tokens: Optional[int]   = 256
    temperature: Optional[float]= 0.7
    stream: Optional[bool]      = False
    stop: Optional[List[str]]   = None

# ── Prompt Formatters ─────────────────────────────────────────────────────────
def format_chatml(messages: List[Message]) -> str:
    """ChatML format — works for most modern models."""
    prompt = ""
    for msg in messages:
        prompt += f"<|im_start|>{msg.role}\n{msg.content}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt

def format_mistral(messages: List[Message]) -> str:
    """Mistral instruct format."""
    prompt = "<s>"
    system = next((m.content for m in messages if m.role == "system"), None)
    turns = [m for m in messages if m.role != "system"]
    for i, msg in enumerate(turns):
        if msg.role == "user":
            sys_tag = f"<<SYS>>\n{system}\n<</SYS>>\n\n" if system and i == 0 else ""
            prompt += f"[INST] {sys_tag}{msg.content} [/INST]"
        elif msg.role == "assistant":
            prompt += f" {msg.content}</s><s>"
    return prompt

def format_llama3(messages: List[Message]) -> str:
    """Llama 3 instruct format."""
    prompt = "<|begin_of_text|>"
    for msg in messages:
        prompt += f"<|start_header_id|>{msg.role}<|end_header_id|>\n\n{msg.content}<|eot_id|>"
    prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
    return prompt

def auto_format(model_name: str, messages: List[Message]) -> str:
    """Pick formatter based on model name heuristic."""
    name = model_name.lower()
    if "llama-3" in name or "llama3" in name:
        return format_llama3(messages)
    if "mistral" in name or "mixtral" in name:
        return format_mistral(messages)
    return format_chatml(messages)   # safe default

# ── Helper ────────────────────────────────────────────────────────────────────
def resolve_model(model_name: str) -> tuple:
    """Returns (model_name, Llama instance)."""
    if model_name in ("auto", "", None):
        registry = find_gguf_files()
        if not registry:
            raise HTTPException(status_code=503, detail="No models found in models/")
        model_name = next(iter(registry))
    llm = get_or_load_model(model_name)
    return model_name, llm

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "Local LLM API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_models": list(loaded_models.keys()),
        "available_models": list(find_gguf_files().keys()),
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {"id": name, "object": "model", "owned_by": "local"}
            for name in find_gguf_files()
        ],
    }

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    model_name, llm = resolve_model(req.model)
    prompt = auto_format(model_name, req.messages)

    stop_seqs = req.stop or ["<|im_end|>", "<|eot_id|>", "</s>", "[INST]"]

    kwargs = dict(
        prompt=prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
        top_k=req.top_k,
        repeat_penalty=req.repeat_penalty,
        stop=stop_seqs,
    )

    if req.stream:
        async def event_stream():
            for chunk in llm(**kwargs, stream=True):
                token = chunk["choices"][0]["text"]
                data = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [{"delta": {"content": token}, "finish_reason": None, "index": 0}],
                }
                yield f"data: {json.dumps(data)}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(event_stream(), media_type="text/event-stream")

    output = llm(**kwargs)
    text = output["choices"][0]["text"].strip()

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": output.get("usage", {}),
    }

@app.post("/v1/completions")
async def completions(req: CompletionRequest):
    model_name, llm = resolve_model(req.model)
    output = llm(
        prompt=req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        stop=req.stop,
    )
    return {
        "id": f"cmpl-{uuid.uuid4().hex[:8]}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [{"text": output["choices"][0]["text"], "index": 0, "finish_reason": "stop"}],
        "usage": output.get("usage", {}),
    }

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

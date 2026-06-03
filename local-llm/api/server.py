
import os, json, time, uuid, httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI(title="Local LLM API")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai"

MODELS = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-2.0-flash",
]

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str = "gemini-1.5-flash"
    messages: List[Message]
    max_tokens: Optional[int] = 1000
    temperature: Optional[float] = 0.7
    stream: Optional[bool] = False

@app.get("/")
def root():
    return {"message": "Local LLM API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok", "models": MODELS}

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": m, "object": "model", "owned_by": "google"} for m in MODELS]
    }

@app.post("/v1/chat/completions")
async def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set")

    payload = {
        "model": req.model,
        "messages": [{"role": m.role, "content": m.content} for m in req.messages],
        "max_tokens": req.max_tokens,
        "temperature": req.temperature,
    }

    async with httpx.AsyncClient(timeout=60) as client:
        response = await client.post(
            f"{GEMINI_BASE}/chat/completions",
            headers={"Authorization": f"Bearer {GEMINI_API_KEY}"},
            json=payload
        )

    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.text)

    return response.json()

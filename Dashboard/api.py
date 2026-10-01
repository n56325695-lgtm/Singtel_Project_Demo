"""
FastAPI backend for the chat UI.

This is a NEW file — main.py stays as-is for CLI testing. Run this one
whenever you want the web UI (kb_chat_ui.html) to have something to talk to.

Run with:
    uvicorn Dashboard.api:app --reload

Then open:
    http://localhost:8000/
"""

import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.generate import generate_answer

app = FastAPI(title="KB search API")

# Allow the browser to call this API even if the HTML is opened a
# different way than being served by this same app (safe to leave on
# for local dev; tighten origins before shipping anywhere real).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serves every screenshot under documents/page_images at:
#   http://localhost:8000/images/<subfolder>/<filename>.png
IMAGES_DIR = "documents/page_images"
app.mount("/images", StaticFiles(directory=IMAGES_DIR), name="images")


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_ui():
    """Serves the chat UI itself, so the whole thing runs on one origin."""
    html_path = os.path.join(os.path.dirname(__file__), "kb_chat_ui.html")
    return FileResponse(html_path)


@app.post("/query")
def query(req: QueryRequest):
    """
    Calls your existing RAG pipeline and reshapes the result into what
    kb_chat_ui.html expects:
        { "answer": str, "images": [ {image_path, caption, page_num, score} ] }

    FIXED: generate_answer() returns "sources" as a flat list of filename
    STRINGS (e.g. ["singtel_doc.pdf"]), not chunk dicts — that key has no
    content_type/image_path/caption to read. The actual per-chunk payload
    data (with content_type, image_path, caption, page_num) lives under
    "retrieved_chunks". Reading "sources" here crashed with
    AttributeError: 'str' object has no attribute 'get' on any non-empty
    response.
    """
    result = generate_answer(req.query)

    answer = result.get("answer", "") if isinstance(result, dict) else str(result)

    chunks = result.get("retrieved_chunks", []) if isinstance(result, dict) else []

    images = []
    seen_paths = set()
    for c in chunks:
        # Only surface actual screenshots, not plain text_chunk hits.
        if not isinstance(c, dict) or c.get("content_type") != "page_image":
            continue
        image_path = c.get("image_path", "")
        if not image_path or image_path in seen_paths:
            continue
        seen_paths.add(image_path)

        rel_path = os.path.relpath(image_path, IMAGES_DIR).replace("\\", "/")
        images.append({
            "image_path": f"/images/{rel_path}",
            "caption": c.get("caption"),
            "page_num": c.get("page_num"),
            "score": c.get("score"),
        })

    return {"answer": answer, "images": images}
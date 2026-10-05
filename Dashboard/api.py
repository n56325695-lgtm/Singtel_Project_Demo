"""
FastAPI backend for the chat UI.

This is a NEW file — main.py stays as-is for CLI testing. Run this one
whenever you want the web UI (kb_chat_ui.html) to have something to talk to.

Run with:
    uvicorn Dashboard.api:app --reload

Then open:
    http://localhost:8000/
"""


import logging
import os
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.generate import generate_answer

log = logging.getLogger("uvicorn.error")

app = FastAPI(title="KB search API")

# Allow the browser to call this API even if the HTML is opened a
# different way than being served by this same app (fine for local dev;
# tighten origins before shipping anywhere real).
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

UI_FILE = os.path.join(os.path.dirname(__file__), "index_singtel.html")


class QueryRequest(BaseModel):
    query: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def serve_ui():
    """Serves the chat UI itself, so everything runs on one origin.

    no-store makes the browser re-read the HTML on every refresh, so
    design edits show up immediately during development.
    """
    return FileResponse(UI_FILE, headers={"Cache-Control": "no-store"})


def _to_float(value):
    """Scores can be numpy floats, which FastAPI cannot serialise."""
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@app.post("/query")
def query(req: QueryRequest):
    """
    Calls the existing RAG pipeline and reshapes the result into:
        { "answer": str,
          "images": [ {image_path, caption, page_num, score} ] }

    generate_answer() returns "sources" as a flat list of filename strings,
    so screenshot data is read from "retrieved_chunks" instead.
    """
    try:
        result = generate_answer(req.query)
    except Exception as exc:
        # Full traceback goes to the uvicorn terminal; a short reason goes
        # back to the browser so the chat bubble can show it.
        log.error("generate_answer failed:\n%s", traceback.format_exc())
        return JSONResponse(
            status_code=503,
            content={
                "answer": "The documentation service is temporarily unavailable. Please try again in a moment.",
                "images": [],
                "error": f"{type(exc).__name__}: {exc}",
            },
        )

    if not isinstance(result, dict):
        result = {"answer": str(result), "retrieved_chunks": []}

    if result.get("error"):
        return JSONResponse(
            status_code=503,
            content={
                "answer": result.get("answer", "The documentation service is temporarily unavailable."),
                "images": [],
                "error": result["error"],
            },
        )

    answer = result.get("answer", "")
    chunks = result.get("retrieved_chunks", [])

    images = []
    seen_paths = set()
    for c in chunks:
        # Only surface real screenshots, not plain text_chunk hits.
        if not isinstance(c, dict) or c.get("content_type") != "page_image":
            continue

        image_path = c.get("image_path", "")
        if not image_path or image_path in seen_paths:
            continue
        seen_paths.add(image_path)

        try:
            rel_path = os.path.relpath(image_path, IMAGES_DIR).replace("\\", "/")
        except ValueError:  # e.g. image stored on a different drive
            continue
        if rel_path.startswith(".."):  # outside the served images folder
            continue

        caption = c.get("caption")
        images.append({
            "image_path": f"/images/{rel_path}",
            "caption": str(caption) if caption else None,
            "page_num": c.get("page_num"),
            "score": _to_float(c.get("score")),
        })

    return {"answer": answer, "images": images}
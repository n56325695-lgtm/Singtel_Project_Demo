
# ************************this logic is correct and handle correctly********************************

"""
INGEST LOGIC — matches the flow diagram exactly, no vision model involved.

  Data/image_records.json (Phase 1 output)
        |
        v
  Only records whose image file SURVIVED manual pruning are kept
  (pruning deletes junk PNGs from the folder; we check disk, not a
  vision-progress file, since there is no vision phase anymore)
        |
        v
  Embed text = caption + extracted_text, per image, NO vision model
        |
        v
  Index in Qdrant, payload per image:
      image_path, page_num, extracted_text, caption, source
        |
        v
  Same collection also holds text/markdown chunks (content_type
  distinguishes the two at query time), exactly as before.

Two bugs fixed from the previous version:
  1. PROGRESS_FILE / vision_data no longer referenced — removed entirely,
     since there is no vision-enrichment phase in this pipeline anymore.
  2. RECORDS_FILE now matches the exact path the extraction script writes
     to ("Data/image_records.json", capital D) — the previous version read
     from "data/image_records.json" (lowercase), which is a guaranteed
     FileNotFoundError on any case-sensitive filesystem (i.e. Linux/prod).
"""

import os
import json
import uuid

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    EMBEDDING_MODEL, EMBEDDING_DIM,
    CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR
)

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# Must match exactly what app/extract_images.py writes to (see RECORDS_FILE there).
RECORDS_FILE = os.path.join("Data", "image_records.json")


def embed_texts(texts: list[str]) -> list[list[float]]:
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return response["embeddings"]


def create_collection():
    client.recreate_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
    )


# --- text/markdown path stays exactly as before ---
def chunk_text(text: str, max_chars: int = 500, min_chars: int = 50) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buffer = [], ""
    for para in paragraphs:
        if len(buffer) + len(para) < max_chars:
            buffer += (" " if buffer else "") + para
        else:
            if len(buffer) >= min_chars:
                chunks.append(buffer)
                buffer = para
            else:
                buffer += " " + para
                chunks.append(buffer)
                buffer = ""
    if buffer:
        chunks.append(buffer)
    return chunks


def load_text_documents(directory: str) -> list[dict]:
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith((".txt", ".md")):
            with open(os.path.join(directory, filename), "r", encoding="utf-8") as f:
                docs.append({"source": filename, "text": f.read()})
    return docs


def build_text_chunk_points(directory: str) -> tuple[list[PointStruct], dict]:
    docs = load_text_documents(directory)
    points, chunk_map = [], {}
    for doc in docs:
        chunks = chunk_text(doc["text"], max_chars=CHUNK_SIZE, min_chars=CHUNK_OVERLAP)
        embeddings = embed_texts(chunks)
        for chunk, embedding in zip(chunks, embeddings):
            chunk_id = str(uuid.uuid4())
            points.append(PointStruct(
                id=chunk_id, vector=embedding,
                payload={"content_type": "text_chunk", "text": chunk, "source": doc["source"]},
            ))
            chunk_map[chunk_id] = {"text": chunk, "source": doc["source"]}
    return points, chunk_map


# --- image path: reads Phase 1 output only, no vision model ---
def build_page_image_points() -> tuple[list[PointStruct], dict]:
    with open(RECORDS_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    # Only records whose image file is still on disk survive here — this is
    # how "manually delete junk images" pruning takes effect. No vision
    # progress file to check anymore.
    ready = [r for r in records if os.path.exists(r["image_path"])]
    skipped = len(records) - len(ready)
    print(f"Building points for {len(ready)} images "
          f"({skipped} skipped — pruned from disk)...")

    texts = []
    for r in ready:
        combined = " ".join(filter(None, [r["caption"], r["extracted_text"]])).strip()
        texts.append(combined if combined else " ")

    embeddings = embed_texts(texts)

    points, image_map = [], {}
    for rec, embedding, text in zip(ready, embeddings, texts):
        point_id = str(uuid.uuid4())

        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={
                "content_type": "page_image",
                "source": rec["source"],
                "page_num": rec["page_num"],
                "image_path": rec["image_path"],
                "extracted_text": rec["extracted_text"],
                "caption": rec["caption"],
            },
        ))
        image_map[point_id] = {
            "text": text,
            "source": rec["source"],
            "page_num": rec["page_num"],
            "image_path": rec["image_path"],
            "caption": rec["caption"],
        }

    return points, image_map


def ingest():
    print("Creating collection...")
    create_collection()

    print("Loading text documents (.txt / .md)...")
    text_points, chunk_map = build_text_chunk_points(DOCUMENTS_DIR)

    print("Loading pruned images...")
    page_points, image_map = build_page_image_points()

    all_points = text_points + page_points
    print(f"Uploading {len(all_points)} points to Qdrant "
          f"({len(text_points)} text chunks, {len(page_points)} images)...")
    client.upsert(collection_name=COLLECTION_NAME, points=all_points)

    combined_map = {**chunk_map, **image_map}
    os.makedirs("Evals/chunks", exist_ok=True)
    with open("Evals/chunks/chunk_map.json", "w", encoding="utf-8") as f:
        json.dump(combined_map, f, ensure_ascii=False, indent=2)

    print(f"Saved chunk_map.json with {len(combined_map)} entries "
          f"({len(chunk_map)} text chunks, {len(image_map)} images)")
    print("Done.")


if __name__ == "__main__":
    ingest()











































# """
# Ingestion for the knowledge base. Handles two kinds of source material,
# both landing in the SAME Qdrant collection (same embedding model, same
# vector space), distinguished by a `content_type` field in the payload:

#   1. Plain text/markdown docs -> paragraph-chunked -> one vector per chunk
#      content_type = "text_chunk"

#   2. Text-native PDFs (guides, dashboards w/ screenshots) -> one vector
#      per EXTRACTED EMBEDDED IMAGE, using the PDF's own text layer (no OCR
#      needed) plus the nearest figure caption found near each screenshot,
#      PLUS a vision-model pass on the screenshot itself (image_derived_
#      description + image_ocr_text) so text baked into the pixels — button
#      labels, table values, status banners — isn't invisible to retrieval.
#      That individual screenshot (not the whole page) is kept on disk for
#      use at answer time.
#      content_type = "page_image"

# Both paths return a map of point_id -> eval-relevant data, keyed with the
# SAME ids used in the Qdrant points, so eval scripts can look up exactly
# what a retrieved id corresponds to.

# Put .txt/.md files and .pdf files together in DOCUMENTS_DIR; this script
# sorts them by extension automatically.
# """

# import os
# import re
# import json
# import uuid
# from pathlib import Path

# import fitz  # PyMuPDF
# import ollama
# from qdrant_client import QdrantClient
# from qdrant_client.models import Distance, VectorParams, PointStruct

# from app.config import (
#     QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
#     EMBEDDING_MODEL, EMBEDDING_DIM,
#     CHUNK_SIZE, CHUNK_OVERLAP, DOCUMENTS_DIR
# )
# from app.vision_extractor import extract_image_info  # NEW

# client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# # ---------------------------------------------------------------------------
# # PDF-specific config (add these to app/config.py once you're happy with them)
# # ---------------------------------------------------------------------------
# PAGE_IMAGES_DIR = os.environ.get("PAGE_IMAGES_DIR", "documents/page_images")
# MIN_IMG_SIZE = int(os.environ.get("MIN_IMG_SIZE", "100"))            # px, skip logos/icons
# CAPTION_PROXIMITY = int(os.environ.get("CAPTION_PROXIMITY", "150"))  # points, max distance to link a caption
# CAPTION_PREFIXES = ("figure", "fig.", "fig ")


# # ---------------------------------------------------------------------------
# # Shared: embedding
# # ---------------------------------------------------------------------------
# def embed_texts(texts: list[str]) -> list[list[float]]:
#     """Generate embeddings using Ollama (bge-m3)."""
#     response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
#     return response["embeddings"]


# def create_collection():
#     """Create (or recreate) the Qdrant collection with the right vector size."""
#     client.recreate_collection(
#         collection_name=COLLECTION_NAME,
#         vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
#     )


# # ---------------------------------------------------------------------------
# # Path 1: plain text/markdown -> chunked -> text_chunk points
# # ---------------------------------------------------------------------------
# def chunk_text(text: str, max_chars: int = 500, min_chars: int = 50) -> list[str]:
#     """Split text into paragraph-based chunks with sane character bounds."""
#     paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

#     chunks = []
#     buffer = ""

#     for para in paragraphs:
#         if len(buffer) + len(para) < max_chars:
#             buffer += (" " if buffer else "") + para
#         else:
#             if len(buffer) >= min_chars:
#                 chunks.append(buffer)
#                 buffer = para
#             else:
#                 buffer += " " + para
#                 chunks.append(buffer)
#                 buffer = ""

#     if buffer:
#         chunks.append(buffer)

#     return chunks


# def load_text_documents(directory: str) -> list[dict]:
#     """Read all .txt and .md files in the documents folder."""
#     docs = []
#     for filename in os.listdir(directory):
#         if filename.endswith((".txt", ".md")):
#             path = os.path.join(directory, filename)
#             with open(path, "r", encoding="utf-8") as f:
#                 docs.append({"source": filename, "text": f.read()})
#     return docs


# def build_text_chunk_points(directory: str) -> tuple[list[PointStruct], dict]:
#     """Chunk + embed every .txt/.md file in directory. Returns the Qdrant
#     points plus a chunk_map (id -> text/source) for eval scripts."""
#     docs = load_text_documents(directory)
#     points = []
#     chunk_map = {}

#     for doc in docs:
#         chunks = chunk_text(doc["text"], max_chars=CHUNK_SIZE, min_chars=CHUNK_OVERLAP)
#         print(f"[{doc['source']}] {len(chunks)} chunks")

#         embeddings = embed_texts(chunks)

#         for chunk, embedding in zip(chunks, embeddings):
#             chunk_id = str(uuid.uuid4())

#             points.append(
#                 PointStruct(
#                     id=chunk_id,
#                     vector=embedding,
#                     payload={
#                         "content_type": "text_chunk",
#                         "text": chunk,
#                         "source": doc["source"],
#                     },
#                 )
#             )
#             chunk_map[chunk_id] = {"text": chunk, "source": doc["source"]}

#     return points, chunk_map


# # ---------------------------------------------------------------------------
# # Path 2: text-native PDFs -> one vector per EXTRACTED IMAGE -> page_image points
# # ---------------------------------------------------------------------------
# def is_blue_caption(span) -> bool:
#     """True if this text span is styled blue (the doc's caption convention)."""
#     color = span["color"]
#     r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
#     return b > 120 and b > r + 30 and b > g + 30



# def get_caption_lines(page: "fitz.Page") -> list[dict]:
#     """Lines that look like captions: either 'Figure:'-style prefix OR blue styling."""
#     captions = []
#     for block in page.get_text("dict")["blocks"]:
#         if block.get("type") != 0:
#             continue
#         for line in block["lines"]:
#             spans = line["spans"]
#             full_text = "".join(s["text"] for s in spans).strip()
#             if not full_text:
#                 continue
#             is_prefixed = full_text.lower().startswith(CAPTION_PREFIXES)
#             is_blue = any(is_blue_caption(s) for s in spans)
#             if is_prefixed or is_blue:
#                 captions.append({"text": full_text, "bbox": line["bbox"]})
#     return captions




# def sanitize_filename(text: str, max_len: int = 60) -> str:
#     text = re.sub(r"^fig(ure)?\s*[:.]?\s*", "", text, flags=re.IGNORECASE)
#     text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
#     return text[:max_len] if text else "untitled"


# def process_pdf_pages(pdf_path: str, output_dir: str) -> list[dict]:
#     """For each page: extract each EMBEDDED screenshot separately (not the
#     whole page), read the page's native text layer, link the nearest
#     caption to each extracted image by vertical proximity, and run a
#     vision model on the screenshot itself to pull out what's IN the
#     image (description + any on-screen text)."""
#     os.makedirs(output_dir, exist_ok=True)
#     stem = Path(pdf_path).stem
#     doc = fitz.open(pdf_path)

#     image_records = []

#     for i, page in enumerate(doc, start=1):
#         page_text = page.get_text().strip()
#         captions = get_caption_lines(page)

#         for img in page.get_images(full=True):
#             xref = img[0]
#             try:
#                 img_bbox = page.get_image_bbox(img)
#             except Exception:
#                 continue

#             pix = fitz.Pixmap(doc, xref)
#             if pix.n - pix.alpha > 3:
#                 pix = fitz.Pixmap(fitz.csRGB, pix)
#             if pix.width < MIN_IMG_SIZE or pix.height < MIN_IMG_SIZE:
#                 pix = None
#                 continue

#             img_center_y = (img_bbox[1] + img_bbox[3]) / 2
#             best_caption, best_dist = None, float("inf")
#             for cap in captions:
#                 cap_center_y = (cap["bbox"][1] + cap["bbox"][3]) / 2
#                 dist = abs(cap_center_y - img_center_y)
#                 if dist < best_dist and dist < CAPTION_PROXIMITY:
#                     best_dist, best_caption = dist, cap["text"]

#             name_part = sanitize_filename(best_caption) if best_caption else f"page{i:03d}_img{xref}"
#             image_path = os.path.join(output_dir, f"{stem}-{name_part}.png")
#             counter = 1
#             while os.path.exists(image_path):
#                 image_path = os.path.join(output_dir, f"{stem}-{name_part}_{counter}.png")
#                 counter += 1

#             pix.save(image_path)
#             pix = None

#             # NEW: ask a vision model what's actually IN this screenshot —
#             # this is the piece the native text layer can never provide.
#             print(f"  [{stem}] page {i}: running vision model on {os.path.basename(image_path)}...")
#             vision_info = extract_image_info(image_path)

#             # NEW: no text found on the image at all (logos, decorative
#             # graphics, icon patterns) -> discard it entirely. Delete the
#             # file from disk and don't create a record/point for it, since
#             # it carries no retrievable information.
#             if not vision_info["image_ocr_text"]:
#                 print(f"  [{stem}] page {i}: no text found, discarding {os.path.basename(image_path)}")
#                 try:
#                     os.remove(image_path)
#                 except OSError as e:
#                     print(f"  [{stem}] page {i}: could not delete {image_path}: {e}")
#                 continue

#             image_records.append({
#                 "page_num": i,
#                 "image_path": image_path,
#                 "extracted_text": page_text,
#                 "caption": best_caption,
#                 "image_derived_description": vision_info["image_derived_description"],  # NEW
#                 "image_ocr_text": vision_info["image_ocr_text"],                          # NEW
#             })

#     doc.close()
#     return image_records


# def build_page_image_points(pdf_dir: str) -> tuple[list[PointStruct], dict]:
#     """Extract + link captions + run vision model + embed + build Qdrant
#     points, one point per EXTRACTED IMAGE (not one per page) for every
#     PDF in pdf_dir. Returns points plus an image_map (id -> text/source/
#     image_path/etc.) keyed with the SAME ids used in the Qdrant points,
#     so eval scripts can look up exactly what a retrieved id corresponds to."""
#     pdf_files = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
#     if not pdf_files:
#         return [], {}

#     points = []
#     image_map = {}

#     for fname in pdf_files:
#         pdf_path = os.path.join(pdf_dir, fname)
#         source = fname
#         out_dir = os.path.join(PAGE_IMAGES_DIR, Path(fname).stem)

#         print(f"[{source}] extracting embedded images + native text layer + vision info...")
#         image_records = process_pdf_pages(pdf_path, out_dir)

#         print(f"[{source}] embedding {len(image_records)} images (bge-m3, one vector each)...")
#         # NEW: caption + page text + vision-model description + vision-model
#         # OCR text all go into the same embedded string, so nothing baked
#         # into the screenshot's pixels is invisible to retrieval.
#         texts = [
#             " ".join(filter(None, [
#                 r["caption"],
#                 r["extracted_text"],
#                 r["image_derived_description"],
#                 r["image_ocr_text"],
#             ])) or " "
#             for r in image_records
#         ]
#         embeddings = embed_texts(texts)

#         for rec, embedding, text in zip(image_records, embeddings, texts):
#             point_id = str(uuid.uuid4())

#             points.append(
#                 PointStruct(
#                     id=point_id,
#                     vector=embedding,
#                     payload={
#                         "content_type": "page_image",
#                         "source": source,
#                         "page_num": rec["page_num"],
#                         "image_path": rec["image_path"],
#                         "extracted_text": rec["extracted_text"],
#                         "caption": rec["caption"],
#                         "image_derived_description": rec["image_derived_description"],  # NEW
#                         "image_ocr_text": rec["image_ocr_text"],                          # NEW
#                     },
#                 )
#             )

#             # same id as the Qdrant point, same "text"/"source" shape as
#             # chunk_map's text entries, plus image-specific fields
#             image_map[point_id] = {
#                 "text": text,
#                 "source": source,
#                 "page_num": rec["page_num"],
#                 "image_path": rec["image_path"],
#                 "caption": rec["caption"],
#                 "image_derived_description": rec["image_derived_description"],  # NEW
#                 "image_ocr_text": rec["image_ocr_text"],                          # NEW
#             }

#     return points, image_map


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------
# def ingest():
#     print("Creating collection...")
#     create_collection()

#     print("Loading text documents (.txt / .md)...")
#     text_points, chunk_map = build_text_chunk_points(DOCUMENTS_DIR)

#     print("Loading PDFs (.pdf)...")
#     page_points, image_map = build_page_image_points(DOCUMENTS_DIR)

#     all_points = text_points + page_points
#     print(f"Uploading {len(all_points)} points to Qdrant "
#           f"({len(text_points)} text chunks, {len(page_points)} images)...")
#     client.upsert(collection_name=COLLECTION_NAME, points=all_points)

#     # merge both maps into one file, keyed by the same ids used in Qdrant,
#     # so eval scripts can resolve ANY retrieved id (text chunk or image)
#     combined_map = {**chunk_map, **image_map}

#     os.makedirs("Evals/chunks", exist_ok=True)
#     with open("Evals/chunks/chunk_map.json", "w", encoding="utf-8") as f:
#         json.dump(combined_map, f, ensure_ascii=False, indent=2)

#     print(f"Saved chunk_map.json with {len(combined_map)} entries "
#           f"({len(chunk_map)} text chunks, {len(image_map)} images)")
#     print("Done.")


# if __name__ == "__main__":
#     ingest()



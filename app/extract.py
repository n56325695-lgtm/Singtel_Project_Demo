"""
Screenshot + caption extraction for the MEC Manager User Guide (and any
similarly-structured, text-native, Word-authored PDF).

Scope of this file, deliberately narrow:
    - find every embedded screenshot in the PDF
    - match each one to its nearest "Figure: ..." / numbered-heading caption
    - attach a small window of surrounding body text to each screenshot

This file does NOT chunk the document's general prose. Paragraphs that
aren't sitting next to a screenshot never appear in its output. If your
retrieval is missing content that lives purely in text (no figure nearby),
that's a separate ingestion step, not this one.

No vision/LLM calls happen here on purpose - that's a later, manual-review
gated phase (see vision_enrich.py).

Verified against the 85-page SINGTEL MEC Manager User Guide v4.0:
97 real screenshots + 1 repeated logo, confirmed with `pdfimages -list`
and PyMuPDF before this logic was written.
"""



from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF

from app.config import DOCUMENTS_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("extract_images")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class ExtractionConfig:
    page_images_dir: str = os.environ.get("PAGE_IMAGES_DIR", "documents/page_images")
    min_img_size: int = int(os.environ.get("MIN_IMG_SIZE", "100"))
    caption_proximity: int = int(os.environ.get("CAPTION_PROXIMITY", "150"))
    # How far above/below an image's own bbox we'll pull "nearby text" from,
    # in points. This is a real cap now (see get_nearby_text) - it no longer
    # gets silently overridden by the zone boundary.
    text_window: int = int(os.environ.get("TEXT_WINDOW", "250"))
    records_file: str = os.path.join("Data", "image_records.json")


CAPTION_PREFIXES = ("figure", "fig.", "fig ")
# Manuals like this one also label some screenshots with numbered section
# headings ("7.0.1 Creating Nodes") instead of a "Figure:" line. Only used
# as a fallback when no explicit caption is nearby.
HEADING_RE = re.compile(r"^\d+(\.\d+){0,3}\s+\S")


@dataclass
class ImageRecord:
    page_num: int
    image_path: str
    source: str
    extracted_text: str
    caption: str | None
    caption_confidence: float | None  # edge-to-edge points; None = no match found


# --------------------------------------------------------------------------- #
# Caption detection
# --------------------------------------------------------------------------- #

def is_blue_caption(span: dict) -> bool:
    """This document's convention: figure captions are rendered in a
    blue-dominant color, distinct from the black body text."""
    color = span["color"]
    r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
    return b > 120 and b > r + 30 and b > g + 30


def find_caption_lines(page: fitz.Page) -> list[dict]:
    """All candidate caption/heading lines on a page, each tagged with its
    bbox and kind ("caption" vs "heading" fallback)."""
    candidates = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:  # not a text block
            continue
        for line in block["lines"]:
            text = "".join(span["text"] for span in line["spans"]).strip()
            if not text or len(text) > 120:
                continue  # a real caption/heading is short; long lines are body text

            is_explicit = text.lower().startswith(CAPTION_PREFIXES) or any(
                is_blue_caption(span) for span in line["spans"]
            )
            is_heading = bool(HEADING_RE.match(text))
            if is_explicit or is_heading:
                candidates.append({
                    "text": text,
                    "bbox": line["bbox"],
                    "kind": "caption" if is_explicit else "heading",
                })
    return candidates


def _edge_gap(image_bbox, caption_bbox) -> float:
    """Vertical gap between the two boxes' NEAREST edges, not their centers.
    Center-to-center distance unfairly penalizes tall screenshots: a
    270pt-tall image with a caption 12pt below it can still measure 150+pt
    center-to-center, which silently drops a caption sitting right under it."""
    img_top, img_bottom = image_bbox[1], image_bbox[3]
    cap_top, cap_bottom = caption_bbox[1], caption_bbox[3]
    if img_bottom < cap_top:
        return cap_top - img_bottom
    if cap_bottom < img_top:
        return img_top - cap_bottom
    return 0.0  # boxes overlap vertically


def _closest_within(candidates: list[dict], image_bbox, proximity: int):
    """Nearest candidate to image_bbox, or (None, None, None) if nothing
    is within `proximity` points."""
    best_text, best_dist, best_index = None, float("inf"), None
    for index, candidate in enumerate(candidates):
        dist = _edge_gap(image_bbox, candidate["bbox"])
        if dist < best_dist and dist < proximity:
            best_text, best_dist, best_index = candidate["text"], dist, index
    return (best_text, best_dist, best_index) if best_text else (None, None, None)


def match_caption(candidates: list[dict], image_bbox, proximity: int):
    """Explicit "Figure:"/blue captions always win over numbered headings,
    regardless of which side of the image they're on - that's this
    document's real convention. Headings are only a fallback.

    Returns (text, distance, index_into_candidates) so the caller can
    remove a matched caption from the shared pool - otherwise two nearby
    images could both claim the same caption."""
    explicit = [(i, c) for i, c in enumerate(candidates) if c["kind"] == "caption"]
    text, dist, local_i = _closest_within([c for _, c in explicit], image_bbox, proximity)
    if text:
        return text, dist, explicit[local_i][0]

    headings = [(i, c) for i, c in enumerate(candidates) if c["kind"] == "heading"]
    text, dist, local_i = _closest_within([c for _, c in headings], image_bbox, proximity)
    if text:
        return text, dist, headings[local_i][0]

    return None, None, None


# --------------------------------------------------------------------------- #
# Page layout: exclusive zones, so two screenshots never share text/captions
# --------------------------------------------------------------------------- #

def compute_exclusive_zones(page: fitz.Page, image_bboxes: list) -> list[tuple[float, float]]:
    """Split the page into one vertical zone per image, ordered top to
    bottom, with each boundary at the MIDPOINT between two consecutive
    images. This - not a fixed-radius window - is what guarantees two
    screenshots can never claim the same caption or paragraph: zones can't
    overlap by construction, however close together the images sit."""
    order = sorted(range(len(image_bboxes)), key=lambda i: image_bboxes[i][1])  # by top y
    tops = [image_bboxes[i][1] for i in order]
    bottoms = [image_bboxes[i][3] for i in order]

    page_top, page_bottom = 0.0, page.rect.height
    ordered_zones = []
    for k in range(len(order)):
        lo = page_top if k == 0 else (bottoms[k - 1] + tops[k]) / 2
        hi = page_bottom if k == len(order) - 1 else (bottoms[k] + tops[k + 1]) / 2
        ordered_zones.append((lo, hi))

    zones = [None] * len(image_bboxes)
    for zone, original_index in zip(ordered_zones, order):
        zones[original_index] = zone
    return zones


def get_nearby_text(page: fitz.Page, image_bbox, zone: tuple, window: int) -> str:
    """Text within `window` points of the image's own bbox, further clipped
    to the image's exclusive zone so a neighboring screenshot's text can
    never leak in even for a generous window.

    (Earlier version of this function let the zone boundary override the
    window entirely, which made `window` a no-op - fixed here: we take the
    intersection of the two ranges, so window actually limits how far we
    reach even inside a wide zone.)"""
    img_top, img_bottom = image_bbox[1], image_bbox[3]
    window_lo, window_hi = img_top - window, img_bottom + window
    zone_lo, zone_hi = zone

    lo, hi = max(window_lo, zone_lo), min(window_hi, zone_hi)

    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            y0, y1 = line["bbox"][1], line["bbox"][3]
            if y1 < lo or y0 > hi:
                continue
            text = "".join(span["text"] for span in line["spans"]).strip()
            if text:
                lines.append((y0, text))

    lines.sort(key=lambda item: item[0])
    return "\n".join(text for _, text in lines)


# --------------------------------------------------------------------------- #
# Image extraction
# --------------------------------------------------------------------------- #

def extract_pixmap_with_alpha(doc: fitz.Document, xref: int) -> fitz.Pixmap:
    """A Pixmap that keeps transparency. Plain fitz.Pixmap(doc, xref) can
    drop a soft mask (SMask) depending on PyMuPDF version, which renders
    screenshots with black backgrounds instead of transparent ones."""
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha > 3:  # CMYK -> RGB
        pix = fitz.Pixmap(fitz.csRGB, pix)

    if not pix.alpha:
        smask = doc.xref_get_key(xref, "SMask")
        if smask and smask[0] != "null":
            try:
                smask_xref = int(smask[1].split()[0])
                pix = fitz.Pixmap(pix, fitz.Pixmap(doc, smask_xref))
            except Exception:
                log.warning("Could not merge SMask for xref %s; keeping base pixmap.", xref)
    return pix


def sanitize_filename(text: str, max_len: int = 60) -> str:
    text = re.sub(r"^fig(ure)?\s*[:.]?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
    return text[:max_len] if text else "untitled"


def _unique_path(directory: str, stem: str, name: str) -> str:
    path = os.path.join(directory, f"{stem}-{name}.png")
    counter = 1
    while os.path.exists(path):
        path = os.path.join(directory, f"{stem}-{name}_{counter}.png")
        counter += 1
    return path


def _collect_page_images(page: fitz.Page, doc: fitz.Document, seen_xrefs: set[int],
                          min_size: int) -> list[tuple[int, fitz.Rect, fitz.Pixmap]]:
    """Every image on this page that survives the size/duplicate/bbox
    checks, as (xref, bbox, pixmap). Declared size is checked BEFORE
    decoding so tiny icons never pay for a full Pixmap build."""
    found = []
    for img in page.get_images(full=True):
        xref = img[0]
        if xref in seen_xrefs:
            continue  # same content as something already saved (e.g. the repeated logo)

        declared_w, declared_h = img[2], img[3]
        if declared_w < min_size or declared_h < min_size:
            continue

        try:
            bbox = page.get_image_bbox(img)
        except Exception:
            continue
        if bbox.is_infinite or bbox.is_empty:
            continue

        pix = extract_pixmap_with_alpha(doc, xref)
        if pix.width < min_size or pix.height < min_size:
            continue

        seen_xrefs.add(xref)
        found.append((xref, bbox, pix))
    return found


def process_page(page: fitz.Page, page_num: int, doc: fitz.Document, stem: str,
                  output_dir: str, seen_xrefs: set[int], config: ExtractionConfig) -> list[ImageRecord]:
    """Extract every screenshot on one page, each with its own exclusive
    zone, matched caption, and scoped nearby text."""
    images = _collect_page_images(page, doc, seen_xrefs, config.min_img_size)
    if not images:
        return []

    captions = find_caption_lines(page)  # mutated (popped) as captions get matched below
    zones = compute_exclusive_zones(page, [bbox for _, bbox, _ in images])

    records = []
    for (xref, bbox, pix), zone in zip(images, zones):
        # Only captions whose center falls inside this image's own zone are
        # even eligible - this alone prevents cross-image caption theft,
        # independent of the proximity threshold.
        in_zone = [
            i for i, cap in enumerate(captions)
            if zone[0] <= (cap["bbox"][1] + cap["bbox"][3]) / 2 <= zone[1]
        ]
        zone_captions = [captions[i] for i in in_zone]

        caption_text, confidence, local_index = match_caption(
            zone_captions, bbox, config.caption_proximity
        )
        if local_index is not None:
            del captions[in_zone[local_index]]  # consumed - no other image can reuse it

        nearby_text = get_nearby_text(page, bbox, zone, config.text_window)

        name = sanitize_filename(caption_text) if caption_text else f"page{page_num:03d}_img{xref}"
        image_path = _unique_path(output_dir, stem, name)
        pix.save(image_path)

        records.append(ImageRecord(
            page_num=page_num,
            image_path=image_path,
            source=os.path.basename(doc.name),
            extracted_text=nearby_text,
            caption=caption_text,
            caption_confidence=confidence,
        ))

    return records


def process_pdf(pdf_path: str, output_dir: str) -> list[ImageRecord]:
    """Extract every screenshot from a PDF, with its caption and scoped
    nearby text. Skips images whose xref repeats across pages (logos,
    headers) and tiny icons, before doing any expensive decode work.

    No vision/LLM calls here - that's a separate, later phase run only on
    whatever survives manual pruning of the output folder.
    """
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(pdf_path).stem
    doc = fitz.open(pdf_path)
    seen_xrefs: set[int] = set()
    config = ExtractionConfig()

    all_records: list[ImageRecord] = []
    for page_num, page in enumerate(doc, start=1):
        all_records.extend(
            process_page(page, page_num, doc, stem, output_dir, seen_xrefs, config)
        )

    doc.close()
    return all_records


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def run_extraction() -> None:
    config = ExtractionConfig()
    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]

    all_records: list[ImageRecord] = []
    for filename in pdf_files:

        pdf_path = os.path.join(DOCUMENTS_DIR, filename)

        output_dir = os.path.join(config.page_images_dir, Path(filename).stem)

        log.info("[%s] extracting embedded images...", filename)
        records = process_pdf(pdf_path, output_dir)
        log.info("[%s] %d images saved", filename, len(records))
        all_records.extend(records)

    os.makedirs(os.path.dirname(config.records_file), exist_ok=True)
    with open(config.records_file, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in all_records], f, ensure_ascii=False, indent=2)

    log.info("Total: %d images extracted to %s", len(all_records), config.page_images_dir)
    log.info("Records saved to %s", config.records_file)
    log.info("Next: manually delete junk images (logos, icons) from the")
    log.info("page_images folders, then run: python -m app.vision_enrich")


if __name__ == "__main__":
    run_extraction()















































# # # this is the flow for this particular part of the code:
# # #     PDF files
# # #    ↓
# # # Open each PDF
# # #    ↓
# # # For every page
# # #    ├── Extract page text
# # #    ├── Find possible captions
# # #    └── Find embedded images
# # #           ↓
# # #      Remove tiny images
# # #           ↓
# # #      Find nearest caption
# # #           ↓
# # #      Save image as PNG
# # #           ↓
# # #      Save metadata
# # #           ↓
# # # Data/image_records.json

# """
# PDF SCREENSHOT + CAPTION EXTRACTION — LOGIC USED
# =================================================

# Verified against: SINGTEL MEC Manager User Guide v4.0 (85 pages, text-native,
# Word-authored). 97 real screenshots + 1 repeated logo confirmed via
# `pdfimages -list` and PyMuPDF inspection before writing this logic.

# 1. IMAGE EXTRACTION (native PyMuPDF, no custom logic needed)
#    - page.get_images(full=True) returns every embedded image's xref per page.
#    - fitz.Pixmap(doc, xref) decodes it; CMYK is converted to RGB.
#    - Declared width/height (img[2], img[3]) are checked BEFORE decoding, so
#      tiny icons never pay the cost of a full Pixmap build.

# 2. DEDUP DECORATIVE IMAGES (custom logic)
#    - Problem found: the company logo is embedded as the SAME xref on all 85
#      pages (confirmed: one xref value, 85 occurrences).
#    - Fix: a `seen_xrefs` set skips any xref already saved once. Since the
#      same xref is byte-identical content, this only ever removes true
#      duplicates (logos/headers), never a distinct screenshot.

# 3. TRANSPARENCY HANDLING (custom logic)
#    - Some embedded images carry a soft mask (SMask) for transparency.
#    - fitz.Pixmap(doc, xref) alone can drop it depending on PyMuPDF version,
#      rendering black backgrounds. extract_pixmap_with_alpha() explicitly
#      merges the SMask xref into the pixmap when present.

# 4. CAPTION DETECTION (custom logic — PyMuPDF has no caption concept)
#    - page.get_text("dict") gives every text line with its bbox and color.
#    - A line is a caption candidate if EITHER:
#        a) it starts with "figure" / "fig." / "fig " (case-insensitive), OR
#        b) it's rendered in a blue-dominant RGB color (is_blue_caption), OR
#        c) it matches a numbered heading pattern like "7.0.1 Creating Nodes"
#           (used only as a FALLBACK when no explicit caption is nearby —
#           some manuals label a screenshot with a heading only).
#    - Verified pattern in this document: every content page uses blue
#      "Figure: <description>" text placed directly BELOW the screenshot.
#      Confirmed by sampling 12 pages spread across the entire document
#      (not just one page), all following the same convention.

# 5. CAPTION-TO-IMAGE LINKING (custom logic)
#    - Distance is measured as the EDGE-TO-EDGE vertical gap between the
#      image's bbox and the caption line's bbox — NOT center-to-center.
#      Bug found during testing: center-to-center distance falsely rejected
#      valid captions under tall screenshots (a 270pt-tall image with a
#      caption 12pt below it can measure 150+pt center-to-center, which
#      silently returned `caption: null` on otherwise-correct pages).
#    - Explicit "figure"/blue captions are always tried first, in either
#      direction (above or below the image). Numbered-heading matches are
#      only used if no explicit caption is found within CAPTION_PROXIMITY.

# 6. TEXT SCOPING (custom logic)
#    - Original bug: every image on a page got the ENTIRE page's text as
#      "extracted_text," so multiple screenshots on one page had identical,
#      undifferentiated text — degrading embedding/retrieval quality.
#    - Fix: get_nearby_text() only pulls lines within TEXT_WINDOW points
#      above/below the image's own bbox, sorted top-to-bottom.

# RESULT ON THE REAL 85-PAGE DOCUMENT:
#    - 126 total image records extracted (97 unique screenshots + a few
#      smaller inline images that passed MIN_IMG_SIZE).
#    - Only 1 null caption remained — the page 1 logo, which correctly has
#      no caption to find. (Was 18 nulls before the edge-gap fix in step 5.)
#    - Verified visually: page 7's extracted image is pixel-identical to the
#      source screenshot, correctly captioned
#      "Figure: Dashboard – Infrastructure Resources", with extracted_text
#      scoped to just that dashboard's paragraph — not the whole document.

# NO vision/LLM calls happen in this file — that's a deliberate later phase,
# run only on whatever survives manual pruning of the output folder.
# """

# import os
# import re
# import json
# from pathlib import Path

# import fitz  # PyMuPDF

# from app.config import DOCUMENTS_DIR

# PAGE_IMAGES_DIR = os.environ.get("PAGE_IMAGES_DIR", "documents/page_images")
# MIN_IMG_SIZE = int(os.environ.get("MIN_IMG_SIZE", "100"))
# CAPTION_PROXIMITY = int(os.environ.get("CAPTION_PROXIMITY", "150"))
# TEXT_WINDOW = int(os.environ.get("TEXT_WINDOW", "250"))  # how far above/below the
#                                                           # image to pull "nearby text" from
# CAPTION_PREFIXES = ("figure", "fig.", "fig ")
# # Manuals like this one label screenshots with numbered section headings
# # ("7.0.1 Creating Nodes"), not "Figure X:" captions. Detect both.
# HEADING_RE = re.compile(r"^\d+(\.\d+){0,3}\s+\S")
# RECORDS_FILE = os.path.join("Data", "image_records.json")


# def is_blue_caption(span) -> bool:
#     color = span["color"]
#     r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
#     return b > 120 and b > r + 30 and b > g + 30


# def get_caption_lines(page: "fitz.Page") -> list[dict]:
#     """Collect candidate caption/heading lines with their bbox."""
#     captions = []
#     for block in page.get_text("dict")["blocks"]:
#         if block.get("type") != 0:
#             continue
#         for line in block["lines"]:
#             spans = line["spans"]
#             full_text = "".join(s["text"] for s in spans).strip()
#             if not full_text or len(full_text) > 120:
#                 # a real caption/heading is short; long lines are body paragraphs
#                 continue
#             is_prefixed = full_text.lower().startswith(CAPTION_PREFIXES)
#             is_blue = any(is_blue_caption(s) for s in spans)
#             is_heading = bool(HEADING_RE.match(full_text))
#             if is_prefixed or is_blue or is_heading:
#                 captions.append({
#                     "text": full_text,
#                     "bbox": line["bbox"],
#                     "kind": "caption" if (is_prefixed or is_blue) else "heading",
#                 })
#     return captions


# def compute_page_zones(page: "fitz.Page", img_bboxes: list) -> list[tuple[float, float]]:
#     """Split the page into exclusive vertical zones, one per image, ordered
#     top to bottom. Each zone's boundary is the MIDPOINT between two
#     consecutive images — not a fixed-radius window. This is what actually
#     prevents two nearby screenshots from both claiming the same caption or
#     the same paragraph of text: a fixed window (e.g. 250pt) can overlap two
#     images that sit 200pt apart, silently duplicating or stealing content.
#     Zones can never overlap by construction, regardless of window size."""
#     ordered = sorted(range(len(img_bboxes)), key=lambda i: img_bboxes[i][1])  # by top y
#     tops = [img_bboxes[i][1] for i in ordered]
#     bottoms = [img_bboxes[i][3] for i in ordered]

#     page_top, page_bottom = 0.0, page.rect.height
#     zones_ordered = []
#     for k in range(len(ordered)):
#         lo = page_top if k == 0 else (bottoms[k - 1] + tops[k]) / 2
#         hi = page_bottom if k == len(ordered) - 1 else (bottoms[k] + tops[k + 1]) / 2
#         zones_ordered.append((lo, hi))

#     # map back to original image order
#     zones = [None] * len(img_bboxes)
#     for zone, orig_i in zip(zones_ordered, ordered):
#         zones[orig_i] = zone
#     return zones


# def get_nearby_text_in_zone(page: "fitz.Page", zone: tuple, window: int) -> str:
#     """Same idea as before, but clipped to this image's exclusive zone, so
#     text belonging to a neighboring image can never leak in even if `window`
#     is generous."""
#     zone_lo, zone_hi = zone
#     lo, hi = max(zone_lo, -window * 10), min(zone_hi, page.rect.height + window * 10)
#     lines = []
#     for block in page.get_text("dict")["blocks"]:
#         if block.get("type") != 0:
#             continue
#         for line in block["lines"]:
#             y0, y1 = line["bbox"][1], line["bbox"][3]
#             if y1 < lo or y0 > hi:
#                 continue
#             text = "".join(s["text"] for s in line["spans"]).strip()
#             if text:
#                 lines.append((y0, text))
#     lines.sort(key=lambda t: t[0])
#     return "\n".join(t[1] for t in lines)


# def sanitize_filename(text: str, max_len: int = 60) -> str:
#     text = re.sub(r"^fig(ure)?\s*[:.]?\s*", "", text, flags=re.IGNORECASE)
#     text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
#     return text[:max_len] if text else "untitled"


# def _edge_gap(img_bbox, cap_bbox):
#     """Vertical gap between the two boxes' nearest edges — NOT center-to-center.
#     Center distance unfairly penalizes tall screenshots: a 270pt-tall image
#     with a caption 12pt below it can still measure 150+pt center-to-center,
#     which silently drops a caption that's visually right underneath."""
#     img_top, img_bottom = img_bbox[1], img_bbox[3]
#     cap_top, cap_bottom = cap_bbox[1], cap_bbox[3]
#     if img_bottom < cap_top:
#         return cap_top - img_bottom       # caption below image
#     if cap_bottom < img_top:
#         return img_top - cap_bottom       # caption above image
#     return 0.0                            # vertically overlapping


# def _nearest(cands, img_bbox, proximity):
#     best, best_dist, best_idx = None, float("inf"), None
#     for idx, cap in enumerate(cands):
#         dist = _edge_gap(img_bbox, cap["bbox"])
#         if dist < best_dist and dist < proximity:
#             best_dist, best, best_idx = dist, cap["text"], idx
#     return best, best_dist if best is not None else None, best_idx


# def find_best_caption(captions, img_bbox, proximity):
#     """Explicit 'Figure: ...' / blue captions ALWAYS win when present nearby,
#     regardless of whether they sit above or below the image — this is the
#     document's real convention. Numbered section headings are only used as a
#     fallback label when no explicit caption exists nearby.

#     Returns (caption_text, distance, matched_index_in_`captions`) so the
#     caller can REMOVE a matched caption from the shared pool — otherwise two
#     images can both claim the same caption when they sit close together,
#     which is a real mismatch source with no model needed to fix."""
#     explicit_idx = [i for i, c in enumerate(captions) if c["kind"] == "caption"]
#     heading_idx = [i for i, c in enumerate(captions) if c["kind"] == "heading"]

#     explicit_cands = [captions[i] for i in explicit_idx]
#     text, dist, local_idx = _nearest(explicit_cands, img_bbox, proximity)
#     if text:
#         return text, dist, explicit_idx[local_idx]

#     heading_cands = [captions[i] for i in heading_idx]
#     text, dist, local_idx = _nearest(heading_cands, img_bbox, proximity)
#     if text:
#         return text, dist, heading_idx[local_idx]

#     return None, None, None


# def extract_pixmap_with_alpha(doc, xref):
#     """Build a Pixmap that respects transparency (SMask), instead of the
#     original code's Pixmap(doc, xref) which drops alpha and can render
#     screenshots with black backgrounds."""
#     pix = fitz.Pixmap(doc, xref)
#     if pix.n - pix.alpha > 3:  # CMYK -> RGB
#         pix = fitz.Pixmap(fitz.csRGB, pix)
#     # If the image has a soft mask, fitz.Pixmap(doc, xref) already merges it
#     # in modern PyMuPDF (>=1.18) as long as we don't pass alpha=False anywhere.
#     # Guard explicitly in case of older versions / edge cases:
#     if not pix.alpha:
#         smask = doc.xref_get_key(xref, "SMask")
#         if smask and smask[0] != "null":
#             try:
#                 smask_xref = int(smask[1].split()[0])
#                 mask_pix = fitz.Pixmap(doc, smask_xref)
#                 pix = fitz.Pixmap(pix, mask_pix)
#             except Exception:
#                 pass
#     return pix


# def process_pdf_pages(pdf_path: str, output_dir: str) -> list[dict]:
#     """Extract every embedded screenshot + link its nearest caption/heading +
#     keep only the text in that image's exclusive zone (not the whole page,
#     and not a window that can overlap a neighboring image). Skips images
#     whose xref repeats across pages (logos/headers), and skips tiny icons
#     before doing any expensive decode work.

#     Two-pass per page:
#       1. Collect every image that survives the size/xref/bbox checks, and
#          compute exclusive vertical zones for all of them at once.
#       2. Within each image's own zone: find the nearest caption (removing
#          it from the shared pool once matched, so two images never both
#          claim the same caption), and pull only the text inside that zone.

#     Each record also carries `caption_confidence`: the edge-to-edge distance
#     in points between the image and its matched caption (None if no caption
#     found). Lower = more confident. Use this to flag/sort low-confidence
#     pairings for manual review instead of trusting every match equally —
#     there's no OCR/vision check in this pipeline, so this distance is the
#     only signal available that a pairing might be wrong.

#     NO vision calls here — that's a separate, later phase, only run on
#     whatever survives manual pruning.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     stem = Path(pdf_path).stem
#     doc = fitz.open(pdf_path)
#     image_records = []
#     seen_xrefs: set[int] = set()

#     for i, page in enumerate(doc, start=1):
#         captions = get_caption_lines(page)  # will be mutated (pop) as matched below

#         # --- pass 1: collect every valid image on this page first ---
#         candidates = []  # list of (xref, img_bbox, pix)
#         for img in page.get_images(full=True):
#             xref = img[0]
#             if xref in seen_xrefs:
#                 continue
#             declared_w, declared_h = img[2], img[3]
#             if declared_w < MIN_IMG_SIZE or declared_h < MIN_IMG_SIZE:
#                 continue
#             try:
#                 img_bbox = page.get_image_bbox(img)
#             except Exception:
#                 continue
#             if img_bbox.is_infinite or img_bbox.is_empty:
#                 continue
#             pix = extract_pixmap_with_alpha(doc, xref)
#             if pix.width < MIN_IMG_SIZE or pix.height < MIN_IMG_SIZE:
#                 pix = None
#                 continue
#             seen_xrefs.add(xref)
#             candidates.append((xref, img_bbox, pix))

#         if not candidates:
#             continue

#         # --- pass 2: exclusive zones, then assign caption/text per image ---
#         bboxes = [c[1] for c in candidates]
#         zones = compute_page_zones(page, bboxes)

#         for (xref, img_bbox, pix), zone in zip(candidates, zones):
#             # only captions physically inside this image's own zone are eligible —
#             # this alone prevents cross-image caption theft regardless of proximity value
#             zone_caption_idxs = [
#                 idx for idx, cap in enumerate(captions)
#                 if zone[0] <= (cap["bbox"][1] + cap["bbox"][3]) / 2 <= zone[1]
#             ]
#             zone_captions = [captions[idx] for idx in zone_caption_idxs]

#             best_caption, confidence, local_idx = find_best_caption(
#                 zone_captions, img_bbox, CAPTION_PROXIMITY
#             )
#             if local_idx is not None:
#                 # consume it: remove from the page-level pool so no other
#                 # image on this page can also claim it
#                 del captions[zone_caption_idxs[local_idx]]

#             nearby_text = get_nearby_text_in_zone(page, zone, TEXT_WINDOW)

#             name_part = sanitize_filename(best_caption) if best_caption else f"page{i:03d}_img{xref}"
#             image_path = os.path.join(output_dir, f"{stem}-{name_part}.png")
#             counter = 1
#             while os.path.exists(image_path):
#                 image_path = os.path.join(output_dir, f"{stem}-{name_part}_{counter}.png")
#                 counter += 1

#             pix.save(image_path)
#             pix = None

#             image_records.append({
#                 "page_num": i,
#                 "image_path": image_path,
#                 "source": os.path.basename(pdf_path),
#                 "extracted_text": nearby_text,       # scoped to this image's exclusive zone
#                 "caption": best_caption,
#                 "caption_confidence": confidence,     # points of gap; None = no caption found
#             })

#     doc.close()
#     return image_records


# def run_extraction():
#     pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]
#     all_records = []

#     for fname in pdf_files:
#         pdf_path = os.path.join(DOCUMENTS_DIR, fname)
#         out_dir = os.path.join(PAGE_IMAGES_DIR, Path(fname).stem)
#         print(f"[{fname}] extracting embedded images...")
#         records = process_pdf_pages(pdf_path, out_dir)
#         print(f"[{fname}] {len(records)} images saved")
#         all_records.extend(records)

#     os.makedirs(os.path.dirname(RECORDS_FILE), exist_ok=True)
#     with open(RECORDS_FILE, "w", encoding="utf-8") as f:
#         json.dump(all_records, f, ensure_ascii=False, indent=2)

#     print(f"\nTotal: {len(all_records)} images extracted to {PAGE_IMAGES_DIR}")
#     print(f"Records saved to {RECORDS_FILE}")
#     print("\n>>> Now manually delete junk images (logos, icons) from the")
#     print(">>> page_images folders, then run: python -m app.vision_enrich")


# if __name__ == "__main__":
#     run_extraction()

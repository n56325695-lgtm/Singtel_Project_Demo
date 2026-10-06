"""
extract.py  --  structure-first extraction of a text-native, Word-authored PDF.

Reads every page ONCE, in reading order, and produces three outputs:
    Data/chunks.json            text + table chunks, each with section path and linked images
    Data/image_records.json     one record per kept screenshot (same keys as the old file, plus more)
    Data/extraction_report.json validation report (headings vs table of contents, skips, tiny chunks...)

Pipeline (each pass is a separate function so a bad answer can be traced to one pass):
    Pass 0  scan_document   body font size, repeated headers/footers, non-content pages, ToC titles
    Pass 1  parse_page      page -> ordered elements: heading / paragraph / table / image
    Pass 2  assign_paths    running heading path (persona > section > subsection > topic)
    Pass 3  build_chunks    chunk inside sections, link images to the chunk they illustrate

No vision / LLM calls happen here.
"""


from __future__ import annotations

import json
import logging
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

try:  # your project config; the fallback only exists so this file runs standalone
    from app.config import DOCUMENTS_DIR
except ImportError:
    DOCUMENTS_DIR = os.environ.get("DOCUMENTS_DIR", "documents")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("extract")


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Config:
    page_images_dir: str = os.environ.get("PAGE_IMAGES_DIR", "documents/page_images")
    data_dir: str = os.environ.get("DATA_DIR", "Data")
    max_chars: int = int(os.environ.get("MAX_CHARS", "1800"))   # ~450 tokens per chunk
    min_chars: int = int(os.environ.get("MIN_CHARS", "300"))    # smaller text chunks get merged forward
    min_img_size: int = int(os.environ.get("MIN_IMG_SIZE", "100"))
    repeated_asset_pages: int = 3        # an image object used on >= this many pages is a logo/decoration
    margin_zone: float = 0.07            # top/bottom 7% of the page = header/footer zone
    context_chars: int = 300             # paragraph text kept before/after each image


CFG = Config()

LIST_RE = re.compile(r"^(\d+[.)]|[a-zA-Z][.)]|[•●▪◦\-–*])\s+")
TOC_LINE_RE = re.compile(r"^(.*?)\s*\.{4,}\s*\d+\s*$")
SENTENCE_END = (".", "!", "?")


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def norm(text: str) -> str:
    """Normalise text: ligatures, NBSP, Word private-use bullets, runs of spaces."""
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\uf0b7", "•").replace("\uf0a7", "•").replace("\u200b", "")
    return re.sub(r"[ \t]+", " ", text).strip()


def key_of(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())


def span_text(line: dict) -> str:
    return norm("".join(s["text"] for s in line["spans"]))


@dataclass
class El:
    """One element of the document, in reading order."""
    kind: str                      # h1 h2 h3 h4 | para | table | image
    page: int
    y0: float
    y1: float
    text: str = ""
    rows: list | None = None       # tables only
    path: tuple = ("", "", "", "")
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Pass 0: whole-document facts
# --------------------------------------------------------------------------- #
def scan_document(doc: fitz.Document, pages: list[dict]) -> dict:
    """Facts that need the whole document:
       - body font size (most common size by character count)
       - which pages are NOT knowledge (cover, version history, table of contents)
       - ToC titles (ground truth to audit heading detection)
       - header/footer lines that repeat on many pages
       - image objects reused on many pages (logos)"""
    sizes = Counter()
    for pd in pages:
        for b in pd["blocks"]:
            for l in b.get("lines", []):
                for s in l["spans"]:
                    if s["text"].strip():
                        sizes[round(s["size"], 1)] += len(s["text"])
    body_size = sizes.most_common(1)[0][0]

    skip, toc_titles = {}, []
    for i, page in enumerate(doc):
        text = page.get_text()
        lines = [norm(x) for x in text.splitlines() if x.strip()]
        toc_hits = [TOC_LINE_RE.match(x) for x in lines]
        toc_hits = [m for m in toc_hits if m]
        if len(toc_hits) >= 3:
            skip[i] = "table_of_contents"
            toc_titles += [m.group(1).strip() for m in toc_hits if m.group(1).strip()]
        elif lines and lines[0].lower().startswith("version control"):
            skip[i] = "version_history"
        elif i == 0 and len(text.strip()) < 200:
            skip[i] = "cover"

    kept = [i for i in range(len(doc)) if i not in skip]
    seen = Counter()
    for i in kept:
        H = pages[i]["height"]
        lines_here = set()
        for b in pages[i]["blocks"]:
            for l in b.get("lines", []):
                y = l["bbox"][1]
                if y < CFG.margin_zone * H or y > (1 - CFG.margin_zone) * H:
                    t = span_text(l)
                    if t:
                        lines_here.add(re.sub(r"\d+", "#", t))
        seen.update(lines_here)
    repeated = {t for t, c in seen.items() if c >= max(3, int(0.3 * len(kept)))}

    xref_pages = Counter()
    for i in kept:
        for xref in {img[0] for img in doc[i].get_images(full=True)}:
            xref_pages[xref] += 1
    repeated_xrefs = {x for x, c in xref_pages.items() if c >= CFG.repeated_asset_pages}

    return {"body_size": body_size, "skip": skip, "toc_titles": toc_titles,
            "repeated_lines": repeated, "repeated_xrefs": repeated_xrefs}


# --------------------------------------------------------------------------- #
# Pass 1: page -> ordered elements
# --------------------------------------------------------------------------- #
def heading_level(spans: list[dict], text: str, body: float) -> int | None:
    """h1..h3 by font size relative to body text; h4 = a short, fully bold line.
    A heading is a WHOLE line, so bold words inside a sentence never qualify."""
    real = [s for s in spans if s["text"].strip()]
    if not real or not text:
        return None
    size = max(s["size"] for s in real)
    bold = all((s["flags"] & 16) or "bold" in s["font"].lower() for s in real)
    if size >= body + 7:
        return 1
    if size >= body + 3.5:
        return 2
    if size >= body + 1.5:
        return 3
    if (bold and len(text) <= 80 and len(text.split()) <= 10
            and not text.endswith((".", ":", ",", ";")) and not LIST_RE.match(text)):
        return 4
    return None



def clean_rows(raw: list[list]) -> list[list[str]]:
    """Merged/spanned cells make PDF tables ragged. Keep only non-empty cells per
    row, and glue wrapped continuation lines (first cell empty, one cell filled)
    back onto the row above."""
    rows = []
    for r in raw:
        cells = [norm((c or "").replace("\n", " ")) for c in r]
        filled = [i for i, c in enumerate(cells) if c and (i == 0 or c != cells[i - 1])]
        if not filled:
            continue
        values = [cells[i] for i in filled]
        if len(values) == 1 and filled[0] > 0 and rows and len(rows[-1]) >= 2:
            rows[-1][-1] += " " + values[0]
        else:
            rows.append(values)
    return rows



def table_width(rows: list[list[str]]) -> int:
    return max((len(r) for r in rows), default=0)



def build_paragraph(lines: list[str]) -> str:
    """Join wrapped lines of ONE block into text. List items (1. / bullets) start
    new lines; ordinary wrapped lines are joined, repairing end-of-line hyphens."""
    items: list[str] = []
    for ln in lines:
        if LIST_RE.match(ln) or not items:
            items.append(ln)
        elif items[-1].endswith("-") and ln[:1].islower():
            items[-1] = items[-1][:-1] + ln
        else:
            items[-1] += " " + ln
    return "\n".join(items)



def save_pixmap(doc: fitz.Document, xref: int, path: str) -> fitz.Pixmap:
    """Pixmap that keeps transparency (soft mask) and is always RGB(A)."""
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha > 3:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if not pix.alpha:
        smask = doc.xref_get_key(xref, "SMask")
        if smask and smask[0] != "null":
            try:
                pix = fitz.Pixmap(pix, fitz.Pixmap(doc, int(smask[1].split()[0])))
            except Exception:
                log.warning("Could not merge SMask for xref %s", xref)
    pix.save(path)
    return pix


def parse_page(doc, page, pd, pno, facts, out_dir, stem, stats) -> list[El]:
    """One page -> elements sorted top to bottom."""
    H, W = pd["height"], pd["width"]
    body = facts["body_size"]
    els: list[El] = []

    # ---- tables first, so their text is not read again as prose ----
    tboxes = []
    for t in page.find_tables().tables:
        rows = clean_rows(t.extract())
        if len(rows) < 2 or table_width(rows) < 2:
            stats["tables_rejected"] += 1       # layout artefact (boxed note, frame, ...)
            continue
        tboxes.append(fitz.Rect(t.bbox))
        els.append(El("table", pno, t.bbox[1], t.bbox[3], rows=rows,
                      meta={"top_ratio": t.bbox[1] / H, "bottom_ratio": t.bbox[3] / H}))

    def in_table(bbox) -> bool:
        cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
        return any(r.x0 <= cx <= r.x1 and r.y0 <= cy <= r.y1 for r in tboxes)

    # ---- text blocks -> headings / paragraphs ----
    for b in pd["blocks"]:
        if b["type"] != 0:
            continue
        para_lines: list[str] = []
        para_y0 = para_y1 = None

        def flush_para():
            nonlocal para_lines, para_y0, para_y1
            if para_lines:
                els.append(El("para", pno, para_y0, para_y1, text=build_paragraph(para_lines)))
            para_lines, para_y0, para_y1 = [], None, None

        for l in b["lines"]:
            text = span_text(l)
            if not text or in_table(l["bbox"]):
                continue
            y = l["bbox"][1]
            in_margin = y < CFG.margin_zone * H or y > (1 - CFG.margin_zone) * H
            if in_margin and (re.sub(r"\d+", "#", text) in facts["repeated_lines"]
                              or re.fullmatch(r"(page\s*)?\d+(\s*of\s*\d+)?", text.lower())):
                stats["margin_lines_dropped"] += 1
                continue
            level = heading_level(l["spans"], text, body)
            if level:
                flush_para()
                prev = els[-1] if els else None
                size = max(s["size"] for s in l["spans"])
                # a long heading wrapped over two lines -> one heading
                if (prev and prev.kind == f"h{level}" and prev.page == pno
                        and l["bbox"][1] - prev.y1 < 0.8 * size):
                    prev.text += " " + text
                    prev.y1 = l["bbox"][3]
                else:
                    els.append(El(f"h{level}", pno, l["bbox"][1], l["bbox"][3], text=text))
            else:
                if para_y0 is None:
                    para_y0 = l["bbox"][1]
                para_y1 = l["bbox"][3]
                para_lines.append(text)
        flush_para()

    # ---- images ----
    n = 0
    for info in page.get_image_info(xrefs=True):
        xref, bbox = info.get("xref", 0), fitz.Rect(info["bbox"])
        w, h = info.get("width", 0), info.get("height", 0)
        reason = None
        if not xref:
            reason = "inline_image"
        elif w < CFG.min_img_size or h < CFG.min_img_size:
            reason = "too_small"
        elif bbox.is_empty or bbox.is_infinite:
            reason = "bad_bbox"
        elif xref in facts["repeated_xrefs"]:
            reason = "repeated_asset"
        elif bbox.width * bbox.height > 0.9 * W * H:
            reason = "full_page_background"
        if reason:
            stats["images_skipped"][reason] += 1
            continue
        n += 1
        if xref in stats["xref_path"]:                 # same screenshot reused on another page:
            path = stats["xref_path"][xref]            # keep this placement, reuse the saved file
            stats["images_reused"] += 1
        else:
            path = os.path.join(out_dir, f"{stem}-p{pno:03d}_{n}.png")
            try:
                save_pixmap(doc, xref, path)
            except Exception as e:  # noqa: BLE001
                log.warning("page %d image xref %s failed: %s", pno, xref, e)
                stats["images_skipped"]["decode_failed"] += 1
                continue
            stats["xref_path"][xref] = path
        els.append(El("image", pno, bbox.y0, bbox.y1, text=path, meta={"uid": f"{pno}-{n}"}))

    els.sort(key=lambda e: (e.y0, e.kind != "table"))
    return els


def build_stream(doc, pages, facts, out_dir, stem, stats) -> list[El]:
    """All pages -> one ordered stream; merges tables that continue across pages."""
    stream: list[El] = []
    for i, page in enumerate(doc):
        if i in facts["skip"]:
            continue
        pno = i + 1
        els = parse_page(doc, page, pages[i], pno, facts, out_dir, stem, stats)
        text_len = sum(len(e.text) for e in els if e.kind in ("para", "h1", "h2", "h3", "h4"))
        stats["page_text_len"][pno] = text_len
        for el in els:
            prev = stream[-1] if stream else None
            if (el.kind == "table" and prev and prev.kind == "table" and prev.page == el.page - 1
                    and prev.meta["bottom_ratio"] > 0.85 and el.meta["top_ratio"] < 0.15
                    and table_width(prev.rows) == table_width(el.rows)):
                rows = el.rows[1:] if el.rows[0] == prev.rows[0] else el.rows   # drop repeated header
                prev.rows += rows
                prev.meta["bottom_ratio"] = el.meta["bottom_ratio"]
                prev.meta["merged_pages"] = prev.meta.get("merged_pages", [prev.page]) + [el.page]
                stats["tables_merged"] += 1
                continue
            stream.append(el)
    return stream


# --------------------------------------------------------------------------- #
# Pass 2: running heading path
# --------------------------------------------------------------------------- #
def assign_paths(stream: list[El]) -> None:
    """A heading replaces its own level and clears the deeper ones. The path is
    carried across page breaks. Level 1 is the persona (PCA / MCA / CTP)."""
    path = ["", "", "", ""]
    for el in stream:
        if el.kind.startswith("h"):
            lvl = int(el.kind[1]) - 1
            path[lvl] = el.text
            for j in range(lvl + 1, 4):
                path[j] = ""
        el.path = tuple(path)


# --------------------------------------------------------------------------- #
# Pass 3: chunking and linking
# --------------------------------------------------------------------------- #
def split_long(text: str, limit: int) -> list[str]:
    """Split one over-long paragraph at line / sentence ends (never mid-word)."""
    if len(text) <= limit:
        return [text]
    parts, cur = [], ""
    for piece in re.split(r"(?<=[.!?])\s+|\n", text):
        while len(piece) > limit:                      # no sentence end at all: hard split on spaces
            cut = piece.rfind(" ", 0, limit) or limit
            parts.append(piece[:cut]); piece = piece[cut:].strip()
        if len(cur) + len(piece) + 1 > limit and cur:
            parts.append(cur); cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur:
        parts.append(cur)
    return parts


def render_table(rows: list[list[str]]) -> str:
    w = table_width(rows)
    rows = [r + [""] * (w - len(r)) for r in rows]
    out = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * w]
    return "\n".join(out + ["| " + " | ".join(r) + " |" for r in rows[1:]])


def build_chunks(stream: list[El], stem: str) -> tuple[list[dict], list[dict]]:
    chunks: list[dict] = []
    images: list[dict] = []

    # --- image context: nearest paragraph before / after, inside the same section ---
    for idx, el in enumerate(stream):
        if el.kind != "image":
            continue
        before = next((e.text for e in reversed(stream[:idx]) if e.kind == "para"), "")
        after = next((e.text for e in stream[idx + 1:]
                      if e.kind == "para" and e.path == el.path), "")
        heads = [p for p in el.path if p]
        images.append({
            "uid": el.meta["uid"], "page_num": el.page, "image_path": el.text, "source": stem + ".pdf",
            "persona": el.path[0], "section_path": " > ".join(heads),
            "caption": " > ".join(heads[-2:]),
            "context_before": before[-CFG.context_chars:],
            "context_after": after[:CFG.context_chars],
            "extracted_text": " ".join(x for x in [before[-CFG.context_chars:], after[:CFG.context_chars]] if x),
            "caption_confidence": None, "chunk_id": None,
        })
    by_uid = {im["uid"]: im for im in images}
    uid_path = {im["uid"]: im["image_path"] for im in images}

    # --- chunking state ---
    cur = {"paras": [], "new": 0, "images": [], "pages": set(), "path": None}

    def new_chunk(kind, path, body, pages, imgs, extra=None):
        c = {"type": kind, "persona": path[0], "path": path,
             "section_path": " > ".join(p for p in path if p),
             "body": body, "pages": sorted(set(pages)), "image_uids": list(imgs),
             "source": stem + ".pdf", **(extra or {})}
        chunks.append(c)
        return c

    def flush():
        if cur["new"] == 0:                        # only overlap text -> nothing new to emit
            if cur["images"] and chunks and chunks[-1]["path"] == cur["path"]:
                chunks[-1]["image_uids"] += cur["images"]
            elif cur["images"]:                    # image-only section: tiny stub, merged forward later
                title = next(p for p in reversed(cur["path"]) if p)
                new_chunk("text_chunk", cur["path"], title, [cur["pages"] and min(cur["pages"]) or 0],
                          cur["images"], {"stub": True})
        else:
            new_chunk("text_chunk", cur["path"], "\n\n".join(cur["paras"]), cur["pages"], cur["images"])
        cur.update(paras=[], new=0, images=[], pages=set())

    def add_para(text, page):
        size = sum(len(p) for p in cur["paras"])
        if cur["paras"] and size + len(text) > CFG.max_chars:
            carry = []
            while len(cur["paras"]) > 1 and cur["paras"][-1].rstrip().endswith(":"):
                carry.insert(0, cur["paras"].pop())          # keep "intro:" with the list that follows
            overlap = cur["paras"][-1:] if cur["new"] > 1 else []
            imgs, path = cur["images"], cur["path"]
            flush()
            cur.update(paras=overlap + carry, new=len(carry), images=[], pages=set(), path=path)
            cur["images"] = []
        cur["paras"].append(text)
        cur["new"] += 1
        cur["pages"].add(page)

    for idx, el in enumerate(stream):
        if el.kind.startswith("h"):
            flush()
            cur["path"] = el.path
            continue
        if cur["path"] != el.path:
            flush()
            cur["path"] = el.path
        if el.kind == "para":
            for piece in split_long(el.text, CFG.max_chars):
                add_para(piece, el.page)
        elif el.kind == "image":
            cur["images"].append(el.meta["uid"])
        elif el.kind == "table":
            lead = ""
            prev = stream[idx - 1] if idx else None
            if prev and prev.kind == "para" and prev.path == el.path and prev.text.rstrip().endswith(":"):
                lead = prev.text
            flush(); cur["path"] = el.path
            header, rest = el.rows[0], el.rows[1:]
            pages = el.meta.get("merged_pages", [el.page])
            part, size, parts = [], 0, []
            for r in rest:
                rlen = sum(len(c) for c in r) + 3 * len(r)
                if part and size + rlen > CFG.max_chars:
                    parts.append(part); part, size = [], 0
                part.append(r); size += rlen
            parts.append(part)
            for n, p in enumerate(parts, 1):
                body = (lead + "\n" if lead else "") + render_table([header] + p)
                new_chunk("table_chunk", el.path, body, pages, [],
                          {"table_part": f"{n}/{len(parts)}", "table_title": lead or el.path[-1] or ""})
    flush()

    # --- merge tiny text chunks: forward (same persona + section), else backward (same subsection) ---
    def join(first, second, title):
        """second.body = [title] + first.body + second.body ; images keep reading order"""
        head = [x for x in [title, "" if first.get("stub") else first["body"]] if x]
        second["body"] = "\n".join(head) + ("\n\n" if head else "") + second["body"]

    merged, i = [], 0
    while i < len(chunks):
        a = chunks[i]
        b = chunks[i + 1] if i + 1 < len(chunks) else None
        prev = merged[-1] if merged else None
        tiny = a["type"] == "text_chunk" and len(a["body"]) < CFG.min_chars
        title = next((p for p in reversed(a["path"]) if p), "")
        if (tiny and b and b["type"] == "table_chunk" and b["path"] == a["path"]
                and b["body"].startswith(a["body"]) and not a["image_uids"]):
            pass                                   # tiny text == the table's own lead-in sentence: drop duplicate
        elif (tiny and b and b["type"] == "text_chunk" and a["path"][:2] == b["path"][:2]
                and (a["path"] == b["path"] or a.get("stub"))        # same topic, or an image-only heading
                and len(a["body"]) + len(b["body"]) <= CFG.max_chars * 1.3):
            join(a, b, "" if a["path"] == b["path"] else title)
            b["image_uids"] = a["image_uids"] + b["image_uids"]
            b["pages"] = sorted(set(a["pages"]) | set(b["pages"]))
        elif (tiny and prev and not a.get("stub")
                and ((prev["type"] == "text_chunk" and prev["path"][:3] == a["path"][:3])
                     or (prev["type"] == "table_chunk" and prev["path"] == a["path"]))
                and len(a["body"]) + len(prev["body"]) <= CFG.max_chars * 1.3):
            head = title if a["path"] != prev["path"] else ""      # keep the topic title when crossing topics
            prev["body"] = (prev["body"] + "\n\n" + "\n".join(x for x in [head, a["body"]] if x)).rstrip()
            prev["image_uids"] += a["image_uids"]
            prev["pages"] = sorted(set(prev["pages"]) | set(a["pages"]))
        else:
            merged.append(a)
        i += 1
    chunks = merged

    for n, c in enumerate(chunks):
        c["id"] = f"{stem}-{n:05d}"
        c["text"] = f"{c['section_path']}\n{c['body']}"          # text that gets embedded
        c.pop("path", None); c.pop("stub", None)
        uids = c.pop("image_uids")
        c["image_paths"] = list(dict.fromkeys(uid_path[u] for u in uids))
        for u in uids:
            by_uid[u]["chunk_id"] = c["id"]
    for im in images:
        im.pop("uid")
    return chunks, images


# --------------------------------------------------------------------------- #
# Validation report
# --------------------------------------------------------------------------- #

def build_report(stream, chunks, images, facts, stats) -> dict:
    detected = [e.text for e in stream if e.kind.startswith("h")]
    have = Counter(key_of(t) for t in detected)
    want = Counter(key_of(t) for t in facts["toc_titles"])
    missing = [t for t in facts["toc_titles"] if want[key_of(t)] > have[key_of(t)]]
    in_toc = set(want)
    lens = [len(c["body"]) for c in chunks]
    return {
        "pages_skipped": {str(k + 1): v for k, v in facts["skip"].items()},
        "body_font_size": facts["body_size"],
        "headings_detected": dict(Counter(e.kind for e in stream if e.kind.startswith("h"))),
        "toc_entries": len(facts["toc_titles"]),
        "toc_entries_missing_in_detected_headings": missing,
        "detected_headings_not_in_toc(h1-h3 only; review)": sorted({
            e.text for e in stream if e.kind in ("h1", "h2", "h3") and key_of(e.text) not in in_toc}),
        "h4_headings_for_review": sorted({e.text for e in stream if e.kind == "h4"}),
        "chunks": {"total": len(chunks), "text": sum(c["type"] == "text_chunk" for c in chunks),
                   "table": sum(c["type"] == "table_chunk" for c in chunks),
                   "per_persona": dict(Counter(c["persona"] for c in chunks))},
        "chunk_chars": {"min": min(lens), "max": max(lens), "avg": sum(lens) // len(lens)},
        "tiny_chunks_under_100_chars": [{"id": c["id"], "path": c["section_path"], "body": c["body"]}
                                        for c in chunks if len(c["body"]) < 100],

        "tables": {"rejected": stats["tables_rejected"], "merged_across_pages": stats["tables_merged"]},
        "images": {"kept": len(images), "same_file_reused_on_another_page": stats["images_reused"], "skipped": dict(stats["images_skipped"]),
                   "without_section": [i["image_path"] for i in images if not i["section_path"]],
                   "not_linked_to_chunk": [i["image_path"] for i in images if not i["chunk_id"]]},
        "margin_lines_dropped": stats["margin_lines_dropped"],
        "low_text_pages(<150 chars; check for missed content)":
            [p for p, n in stats["page_text_len"].items() if n < 150],
    }



# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #

def process_pdf(pdf_path: str) -> tuple[list[dict], list[dict], dict]:
    stem = Path(pdf_path).stem
    out_dir = os.path.join(CFG.page_images_dir, stem)
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages = [{**p.get_text("dict"), "height": p.rect.height, "width": p.rect.width} for p in doc]
    facts = scan_document(doc, pages)
    stats = {"tables_rejected": 0, "tables_merged": 0, "margin_lines_dropped": 0,
             "images_skipped": Counter(), "xref_path": {}, "images_reused": 0, "page_text_len": {}}
    stream = build_stream(doc, pages, facts, out_dir, stem, stats)
    assign_paths(stream)
    chunks, images = build_chunks(stream, stem)
    report = build_report(stream, chunks, images, facts, stats)
    doc.close()
    return chunks, images, report


def run_extraction() -> None:
    os.makedirs(CFG.data_dir, exist_ok=True)
    all_chunks, all_images, reports = [], [], {}
    for fn in sorted(f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")):
        log.info("[%s] extracting...", fn)
        c, i, r = process_pdf(os.path.join(DOCUMENTS_DIR, fn))
        log.info("[%s] %d chunks (%d table), %d images", fn, len(c), r["chunks"]["table"], len(i))
        all_chunks += c; all_images += i; reports[fn] = r
    for name, obj in (("chunks.json", all_chunks), ("image_records.json", all_images),
                      ("extraction_report.json", reports)):
        with open(os.path.join(CFG.data_dir, name), "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s/{chunks,image_records,extraction_report}.json", CFG.data_dir)
    log.info("Next: read extraction_report.json, spot-check, then run ingest.")


if __name__ == "__main__":
    run_extraction()




































































































































# """
# Screenshot + caption extraction for the MEC Manager User Guide (and any
# similarly-structured, text-native, Word-authored PDF).

# Scope of this file, deliberately narrow:
#     - find every embedded screenshot in the PDF
#     - match each one to its nearest "Figure: ..." / numbered-heading caption
#     - attach a small window of surrounding body text to each screenshot

# This file does NOT chunk the document's general prose. Paragraphs that
# aren't sitting next to a screenshot never appear in its output. If your
# retrieval is missing content that lives purely in text (no figure nearby),
# that's a separate ingestion step, not this one.

# No vision/LLM calls happen here on purpose - that's a later, manual-review
# gated phase (see vision_enrich.py).

# Verified against the 85-page SINGTEL MEC Manager User Guide v4.0:
# 97 real screenshots + 1 repeated logo, confirmed with `pdfimages -list`
# and PyMuPDF before this logic was written.
# """



# from __future__ import annotations

# import json
# import logging
# import os
# import re
# from dataclasses import asdict, dataclass
# from pathlib import Path

# import fitz  # PyMuPDF

# from app.config import DOCUMENTS_DIR

# logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
# log = logging.getLogger("extract_images")


# # --------------------------------------------------------------------------- #
# # Config
# # --------------------------------------------------------------------------- #

# @dataclass(frozen=True)
# class ExtractionConfig:
#     page_images_dir: str = os.environ.get("PAGE_IMAGES_DIR", "documents/page_images")
#     min_img_size: int = int(os.environ.get("MIN_IMG_SIZE", "100"))
#     caption_proximity: int = int(os.environ.get("CAPTION_PROXIMITY", "150"))
#     # How far above/below an image's own bbox we'll pull "nearby text" from,
#     # in points. This is a real cap now (see get_nearby_text) - it no longer
#     # gets silently overridden by the zone boundary.
#     text_window: int = int(os.environ.get("TEXT_WINDOW", "250"))
#     records_file: str = os.path.join("Data", "image_records.json")


# CAPTION_PREFIXES = ("figure", "fig.", "fig ")
# # Manuals like this one also label some screenshots with numbered section
# # headings ("7.0.1 Creating Nodes") instead of a "Figure:" line. Only used
# # as a fallback when no explicit caption is nearby.
# HEADING_RE = re.compile(r"^\d+(\.\d+){0,3}\s+\S")


# @dataclass
# class ImageRecord:
#     page_num: int
#     image_path: str
#     source: str
#     extracted_text: str
#     caption: str | None
#     caption_confidence: float | None  # edge-to-edge points; None = no match found


# # --------------------------------------------------------------------------- #
# # Caption detection
# # --------------------------------------------------------------------------- #

# def is_blue_caption(span: dict) -> bool:
#     """This document's convention: figure captions are rendered in a
#     blue-dominant color, distinct from the black body text."""
#     color = span["color"]
#     r, g, b = (color >> 16) & 255, (color >> 8) & 255, color & 255
#     return b > 120 and b > r + 30 and b > g + 30


# def find_caption_lines(page: fitz.Page) -> list[dict]:
#     """All candidate caption/heading lines on a page, each tagged with its
#     bbox and kind ("caption" vs "heading" fallback)."""
#     candidates = []
#     for block in page.get_text("dict")["blocks"]:
#         if block.get("type") != 0:  # not a text block
#             continue
#         for line in block["lines"]:
#             text = "".join(span["text"] for span in line["spans"]).strip()
#             if not text or len(text) > 120:
#                 continue  # a real caption/heading is short; long lines are body text

#             is_explicit = text.lower().startswith(CAPTION_PREFIXES) or any(
#                 is_blue_caption(span) for span in line["spans"]
#             )
#             is_heading = bool(HEADING_RE.match(text))
#             if is_explicit or is_heading:
#                 candidates.append({
#                     "text": text,
#                     "bbox": line["bbox"],
#                     "kind": "caption" if is_explicit else "heading",
#                 })
#     return candidates


# def _edge_gap(image_bbox, caption_bbox) -> float:
#     """Vertical gap between the two boxes' NEAREST edges, not their centers.
#     Center-to-center distance unfairly penalizes tall screenshots: a
#     270pt-tall image with a caption 12pt below it can still measure 150+pt
#     center-to-center, which silently drops a caption sitting right under it."""
#     img_top, img_bottom = image_bbox[1], image_bbox[3]
#     cap_top, cap_bottom = caption_bbox[1], caption_bbox[3]
#     if img_bottom < cap_top:
#         return cap_top - img_bottom
#     if cap_bottom < img_top:
#         return img_top - cap_bottom
#     return 0.0  # boxes overlap vertically


# def _closest_within(candidates: list[dict], image_bbox, proximity: int):
#     """Nearest candidate to image_bbox, or (None, None, None) if nothing
#     is within `proximity` points."""
#     best_text, best_dist, best_index = None, float("inf"), None
#     for index, candidate in enumerate(candidates):
#         dist = _edge_gap(image_bbox, candidate["bbox"])
#         if dist < best_dist and dist < proximity:
#             best_text, best_dist, best_index = candidate["text"], dist, index
#     return (best_text, best_dist, best_index) if best_text else (None, None, None)


# def match_caption(candidates: list[dict], image_bbox, proximity: int):
#     """Explicit "Figure:"/blue captions always win over numbered headings,
#     regardless of which side of the image they're on - that's this
#     document's real convention. Headings are only a fallback.

#     Returns (text, distance, index_into_candidates) so the caller can
#     remove a matched caption from the shared pool - otherwise two nearby
#     images could both claim the same caption."""
#     explicit = [(i, c) for i, c in enumerate(candidates) if c["kind"] == "caption"]
#     text, dist, local_i = _closest_within([c for _, c in explicit], image_bbox, proximity)
#     if text:
#         return text, dist, explicit[local_i][0]

#     headings = [(i, c) for i, c in enumerate(candidates) if c["kind"] == "heading"]
#     text, dist, local_i = _closest_within([c for _, c in headings], image_bbox, proximity)
#     if text:
#         return text, dist, headings[local_i][0]

#     return None, None, None


# # --------------------------------------------------------------------------- #
# # Page layout: exclusive zones, so two screenshots never share text/captions
# # --------------------------------------------------------------------------- #

# def compute_exclusive_zones(page: fitz.Page, image_bboxes: list) -> list[tuple[float, float]]:
#     """Split the page into one vertical zone per image, ordered top to
#     bottom, with each boundary at the MIDPOINT between two consecutive
#     images. This - not a fixed-radius window - is what guarantees two
#     screenshots can never claim the same caption or paragraph: zones can't
#     overlap by construction, however close together the images sit."""
#     order = sorted(range(len(image_bboxes)), key=lambda i: image_bboxes[i][1])  # by top y
#     tops = [image_bboxes[i][1] for i in order]
#     bottoms = [image_bboxes[i][3] for i in order]

#     page_top, page_bottom = 0.0, page.rect.height
#     ordered_zones = []
#     for k in range(len(order)):
#         lo = page_top if k == 0 else (bottoms[k - 1] + tops[k]) / 2
#         hi = page_bottom if k == len(order) - 1 else (bottoms[k] + tops[k + 1]) / 2
#         ordered_zones.append((lo, hi))

#     zones = [None] * len(image_bboxes)
#     for zone, original_index in zip(ordered_zones, order):
#         zones[original_index] = zone
#     return zones


# def get_nearby_text(page: fitz.Page, image_bbox, zone: tuple, window: int) -> str:
#     """Text within `window` points of the image's own bbox, further clipped
#     to the image's exclusive zone so a neighboring screenshot's text can
#     never leak in even for a generous window.

#     (Earlier version of this function let the zone boundary override the
#     window entirely, which made `window` a no-op - fixed here: we take the
#     intersection of the two ranges, so window actually limits how far we
#     reach even inside a wide zone.)"""
#     img_top, img_bottom = image_bbox[1], image_bbox[3]
#     window_lo, window_hi = img_top - window, img_bottom + window
#     zone_lo, zone_hi = zone

#     lo, hi = max(window_lo, zone_lo), min(window_hi, zone_hi)

#     lines = []
#     for block in page.get_text("dict")["blocks"]:
#         if block.get("type") != 0:
#             continue
#         for line in block["lines"]:
#             y0, y1 = line["bbox"][1], line["bbox"][3]
#             if y1 < lo or y0 > hi:
#                 continue
#             text = "".join(span["text"] for span in line["spans"]).strip()
#             if text:
#                 lines.append((y0, text))

#     lines.sort(key=lambda item: item[0])
#     return "\n".join(text for _, text in lines)


# # --------------------------------------------------------------------------- #
# # Image extraction
# # --------------------------------------------------------------------------- #

# def extract_pixmap_with_alpha(doc: fitz.Document, xref: int) -> fitz.Pixmap:
#     """A Pixmap that keeps transparency. Plain fitz.Pixmap(doc, xref) can
#     drop a soft mask (SMask) depending on PyMuPDF version, which renders
#     screenshots with black backgrounds instead of transparent ones."""
#     pix = fitz.Pixmap(doc, xref)
#     if pix.n - pix.alpha > 3:  # CMYK -> RGB
#         pix = fitz.Pixmap(fitz.csRGB, pix)

#     if not pix.alpha:
#         smask = doc.xref_get_key(xref, "SMask")
#         if smask and smask[0] != "null":
#             try:
#                 smask_xref = int(smask[1].split()[0])
#                 pix = fitz.Pixmap(pix, fitz.Pixmap(doc, smask_xref))
#             except Exception:
#                 log.warning("Could not merge SMask for xref %s; keeping base pixmap.", xref)
#     return pix


# def sanitize_filename(text: str, max_len: int = 60) -> str:
#     text = re.sub(r"^fig(ure)?\s*[:.]?\s*", "", text, flags=re.IGNORECASE)
#     text = re.sub(r"[^a-zA-Z0-9_-]+", "_", text).strip("_")
#     return text[:max_len] if text else "untitled"


# def _unique_path(directory: str, stem: str, name: str) -> str:
#     path = os.path.join(directory, f"{stem}-{name}.png")
#     counter = 1
#     while os.path.exists(path):
#         path = os.path.join(directory, f"{stem}-{name}_{counter}.png")
#         counter += 1
#     return path


# def _collect_page_images(page: fitz.Page, doc: fitz.Document, seen_xrefs: set[int],
#                           min_size: int) -> list[tuple[int, fitz.Rect, fitz.Pixmap]]:
#     """Every image on this page that survives the size/duplicate/bbox
#     checks, as (xref, bbox, pixmap). Declared size is checked BEFORE
#     decoding so tiny icons never pay for a full Pixmap build."""
#     found = []
#     for img in page.get_images(full=True):
#         xref = img[0]
#         if xref in seen_xrefs:
#             continue  # same content as something already saved (e.g. the repeated logo)

#         declared_w, declared_h = img[2], img[3]
#         if declared_w < min_size or declared_h < min_size:
#             continue

#         try:
#             bbox = page.get_image_bbox(img)
#         except Exception:
#             continue
#         if bbox.is_infinite or bbox.is_empty:
#             continue

#         pix = extract_pixmap_with_alpha(doc, xref)
#         if pix.width < min_size or pix.height < min_size:
#             continue

#         seen_xrefs.add(xref)
#         found.append((xref, bbox, pix))
#     return found


# def process_page(page: fitz.Page, page_num: int, doc: fitz.Document, stem: str,
#                   output_dir: str, seen_xrefs: set[int], config: ExtractionConfig) -> list[ImageRecord]:
#     """Extract every screenshot on one page, each with its own exclusive
#     zone, matched caption, and scoped nearby text."""
#     images = _collect_page_images(page, doc, seen_xrefs, config.min_img_size)
#     if not images:
#         return []

#     captions = find_caption_lines(page)  # mutated (popped) as captions get matched below
#     zones = compute_exclusive_zones(page, [bbox for _, bbox, _ in images])

#     records = []
#     for (xref, bbox, pix), zone in zip(images, zones):
#         # Only captions whose center falls inside this image's own zone are
#         # even eligible - this alone prevents cross-image caption theft,
#         # independent of the proximity threshold.
#         in_zone = [
#             i for i, cap in enumerate(captions)
#             if zone[0] <= (cap["bbox"][1] + cap["bbox"][3]) / 2 <= zone[1]
#         ]
#         zone_captions = [captions[i] for i in in_zone]

#         caption_text, confidence, local_index = match_caption(
#             zone_captions, bbox, config.caption_proximity
#         )
#         if local_index is not None:
#             del captions[in_zone[local_index]]  # consumed - no other image can reuse it

#         nearby_text = get_nearby_text(page, bbox, zone, config.text_window)

#         name = sanitize_filename(caption_text) if caption_text else f"page{page_num:03d}_img{xref}"
#         image_path = _unique_path(output_dir, stem, name)
#         pix.save(image_path)

#         records.append(ImageRecord(
#             page_num=page_num,
#             image_path=image_path,
#             source=os.path.basename(doc.name),
#             extracted_text=nearby_text,
#             caption=caption_text,
#             caption_confidence=confidence,
#         ))

#     return records


# def process_pdf(pdf_path: str, output_dir: str) -> list[ImageRecord]:
#     """Extract every screenshot from a PDF, with its caption and scoped
#     nearby text. Skips images whose xref repeats across pages (logos,
#     headers) and tiny icons, before doing any expensive decode work.

#     No vision/LLM calls here - that's a separate, later phase run only on
#     whatever survives manual pruning of the output folder.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     stem = Path(pdf_path).stem
#     doc = fitz.open(pdf_path)
#     seen_xrefs: set[int] = set()
#     config = ExtractionConfig()

#     all_records: list[ImageRecord] = []
#     for page_num, page in enumerate(doc, start=1):
#         all_records.extend(
#             process_page(page, page_num, doc, stem, output_dir, seen_xrefs, config)
#         )

#     doc.close()
#     return all_records


# # --------------------------------------------------------------------------- #
# # Entry point
# # --------------------------------------------------------------------------- #

# def run_extraction() -> None:
#     config = ExtractionConfig()
#     pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.lower().endswith(".pdf")]

#     all_records: list[ImageRecord] = []
#     for filename in pdf_files:

#         pdf_path = os.path.join(DOCUMENTS_DIR, filename)

#         output_dir = os.path.join(config.page_images_dir, Path(filename).stem)

#         log.info("[%s] extracting embedded images...", filename)
#         records = process_pdf(pdf_path, output_dir)
#         log.info("[%s] %d images saved", filename, len(records))
#         all_records.extend(records)

#     os.makedirs(os.path.dirname(config.records_file), exist_ok=True)
#     with open(config.records_file, "w", encoding="utf-8") as f:
#         json.dump([asdict(r) for r in all_records], f, ensure_ascii=False, indent=2)

#     log.info("Total: %d images extracted to %s", len(all_records), config.page_images_dir)
#     log.info("Records saved to %s", config.records_file)
#     log.info("Next: manually delete junk images (logos, icons) from the")
#     log.info("page_images folders, then run: python -m app.vision_enrich")


# if __name__ == "__main__":
#     run_extraction()















































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

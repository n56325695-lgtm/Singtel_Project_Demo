"""
generate.py  --  retrieve -> prompt -> LLM -> answer + screenshots.

Your old file, adapted to the new chunk format. What changed and why:

  1. No more page_image chunks. Every retrieved chunk is a text or table chunk and carries
     `image_paths` (a LIST of screenshots). Images are collected from the chunks the LLM says it used.
  2. Chunk labels now include role (persona), section path and pages, so the LLM can tell the
     Private Cloud / Multi Cloud / Cloud Tenant versions of the same topic apart.
  3. figure_description is no longer written by the LLM. The LLM only sees text, never the
     screenshots, so asking it to "describe the figure" invites invented detail. The field is now
     built deterministically from the section path of the images that are shown.
  4. Prompt rewritten for the Paragon guide (role rules, tables, overlapping chunks).
  5. Number of images is capped, duplicates removed, and the "used_context true but bad indices"
     fallback only trusts the top-ranked chunks instead of all of them.
"""
import json

from app.rag import retrieve
from app.llm_client import call_llm

MAX_IMAGES = 6          # screenshots returned per answer
FALLBACK_CHUNKS = 2     # if the LLM gives no usable indices, trust only the top-N retrieved chunks


# this is called system prompt
PROMPT_TEMPLATE = """You are a helpful assistant for the Paragon platform user guide. Answer the question using ONLY the context below as your source of truth.

Context format
Each block is labeled [Chunk N | Role: ... | Section: ... | Pages: ...]. Some retrieved chunks may not be relevant to this question, even though they were retrieved. You must judge relevance yourself.
- The guide has three role-specific parts: Private Cloud Administrator (PCA), Multi Cloud Administrator (MCA) and Cloud Tenant Portal Administrator (CTP). The same topic (for example OS Images or Nodes) is documented separately for each role and can differ.
- Tables appear as markdown tables, usually with a header row such as Field | Description.
- Consecutive chunks of the same section can overlap by one paragraph. Do not repeat overlapping text.

How to answer
- If the question names a role, use only chunks for that role. If it names no role and the relevant chunks come from more than one role, answer for each role separately and say which role each part applies to. Never merge steps from different roles into one procedure.
- If the context describes relevant steps, screens, or information, even if not phrased exactly like the question, use it to give a clear, helpful answer.
- If a procedure spans multiple chunks (e.g. steps 1-6 in one chunk, steps 7-12 in another), combine ALL of them into one complete answer, in order (use the page numbers to order them). Never answer with only the first part.
- If the context is genuinely unrelated to the question, set used_context to false and give a brief, honest reply without inventing details.
- Never invent steps, screens, fields, or details that are not in the context.

Answer style (important)
- Write in complete sentences, never a bare list of words. Start with a one-sentence direct answer to the question.
- Then add the closely related details found in the SAME context that a user would need to act on the answer. Examples: what each value means, where in the UI to find it (menu path), the field or column name, caveats, warnings, prerequisites, limits, or what happens in edge cases.
- Use only details present in the context. Do not pad with generic advice or repeat yourself.
- For procedures, use a numbered list with one step per line. For lists of options or values, use bullets with a short explanation for each when the context provides one.
- Use markdown (**bold** for UI labels and key terms). Inside the JSON string, write line breaks as \\n.
- Typical length: 2-6 sentences for factual questions; as long as needed for procedures.

Example of the expected style
Question: What does the Compute Nodes page show for a Private Cloud Administrator?
Answer: "For a Private Cloud Administrator, the **Compute Nodes** page lists the nodes provisioned within their administrative scope, and it is view-only: the PCA cannot start, stop, edit or delete nodes from this page.\\n\\nThe table shows these fields:\\n- **Hostname**: the hostname assigned to the node.\\n- **Node Size**: the compute resource profile assigned to the node.\\n- **Private IP**, **CIDR**, **OS Image**, **GPU** and **Status**.\\n\\nUse the **Search** field to find a node, and enable **Show Deleted** to include deleted nodes."

Ranking chunks (controls which screenshots are shown first)
relevant_chunk_indices must be ordered from MOST relevant to LEAST relevant, regardless of where each chunk sits in the context. Only include a chunk index if you actually drew on that chunk's content.

Context:
{context}

Question: {question}

Respond with ONLY valid JSON in this exact shape, no other text, no markdown fences:
{{"used_context": true or false, "answer": "your helpful, complete answer in markdown", "relevant_chunk_indices": [chunk numbers ordered from most to least relevant, e.g. [2, 0, 3]]}}"""




def _chunk_text(c: dict) -> str:
    """Chunk text as stored by ingest.py is 'section path\\nbody'. The section path is
    already shown in the chunk label, so drop that first line to avoid saying it twice."""
    text = c.get("text") or "(no text)"
    path = c.get("section_path")
    if path and text.startswith(path + "\n"):
        text = text[len(path) + 1:]
    return text


def _chunk_label(i: int, c: dict) -> str:
    pages = c.get("pages") or ([c["page_num"]] if c.get("page_num") else [])
    page_txt = (f"{pages[0]}-{pages[-1]}" if len(pages) > 1 else str(pages[0])) if pages else "?"
    return (f"[Chunk {i} | Role: {c.get('persona') or 'unknown'} | "
            f"Section: {c.get('section_path') or 'unknown'} | Pages: {page_txt}]")


# building the prompt using this function to format and send to the llm for answer generation.
def build_prompt(question: str, chunks: list[dict]) -> str:
    """Combine retrieved chunks into one numbered context block and fill the template.
    Each chunk is tagged [Chunk N] so the LLM can report exactly which chunks it used."""
    context = "\n\n".join(f"{_chunk_label(i, c)}\n{_chunk_text(c)}" for i, c in enumerate(chunks))
    return PROMPT_TEMPLATE.format(context=context, question=question)


# parse the answer from the llm and return the answer
def _parse_llm_json(raw: str) -> tuple[str, bool, list[int]]:
    """Parse the model's JSON response into (answer, used_context, relevant_chunk_indices)."""
    text = raw.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()

    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    try:
        parsed = json.loads(text)
        answer = parsed.get("answer", raw)
        used_context = bool(parsed.get("used_context", False))
        raw_indices = parsed.get("relevant_chunk_indices", [])
        indices = []
        for i in raw_indices:
            if isinstance(i, (int, float)) and not isinstance(i, bool) and int(i) not in indices:
                indices.append(int(i))                      # keep order, drop duplicates
        return answer, used_context, indices
    # handle the error correctness here in this step
    except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
        return raw, False, []


def _collect_images(used_chunks: list[dict]) -> tuple[list[str], str | None]:
    """Screenshots of the chunks the LLM used, most relevant chunk first, steps in document order
    inside a chunk. Duplicates removed, capped at MAX_IMAGES. Also returns a deterministic
    figure description built from the section path (never from LLM guesses)."""
    paths, seen, described = [], set(), []
    for c in used_chunks:
        imgs = c.get("image_paths") or ([c["image_path"]] if c.get("image_path") else [])
        added = False
        for p in imgs:
            if p in seen or len(paths) >= MAX_IMAGES:
                continue
            seen.add(p); paths.append(p); added = True
        if added and c.get("section_path"):
            page = c.get("page_num")
            described.append(f"{c['section_path']}" + (f" (page {page})" if page else ""))
    description = ("Screenshot(s) from: " + "; ".join(described)) if described else None
    return paths, description


def _error_response(question: str, message: str, error: str) -> dict:
    return {
        "question": question, "answer": message, "sources": [], "citations": [],
        "figure_description": None, "retrieved_chunks": [], "image_path": None,
        "image_paths": [], "used_context": False, "error": error,
    }


def generate_answer(question: str) -> dict:
    """Full RAG pipeline: retrieve relevant chunks, build prompt, call LLM.

    retrieve() already drops low-score hits (score_threshold), so an empty list means nothing
    relevant was indexed. On top of that the LLM reports which chunks it actually used
    (relevant_chunk_indices); only those get their screenshots and citations attached."""
    try:
        chunks = retrieve(question)
    except Exception as exc:
        return _error_response(
            question,
            "The documentation service is temporarily unavailable. Please make sure the vector "
            "search service is running, then try again.",
            f"Documentation search unavailable: {type(exc).__name__}")

    if not chunks:
        prompt = PROMPT_TEMPLATE.format(context="No relevant documentation found.", question=question)
    else:
        prompt = build_prompt(question, chunks)

    try:
        raw = call_llm(prompt)
    except Exception as exc:
        return _error_response(
            question,
            "The AI answer service is temporarily unavailable. Please try again in a moment.",
            f"LLM service unavailable: {type(exc).__name__}")

    answer, used_context, relevant_indices = _parse_llm_json(raw)
    used_context = used_context and bool(chunks)     # nothing retrieved -> the answer cannot be grounded

    image_paths, figure_description = [], None
    sources, citations, relevant_chunks = [], [], []

    if used_context and chunks:
        # only keep the chunks the model confirmed it used, not everything retrieved
        used_chunks = [chunks[i] for i in relevant_indices if 0 <= i < len(chunks)]
        # model said used_context=true but gave no usable indices: trust only the best-scored chunks
        if not used_chunks:
            used_chunks = chunks[:FALLBACK_CHUNKS]

        relevant_chunks = used_chunks
        sources = list(dict.fromkeys(c.get("source", "unknown") for c in used_chunks))
        citations = [{"source": c.get("source"), "role": c.get("persona"),
                      "section_path": c.get("section_path"), "pages": c.get("pages")}
                     for c in used_chunks]
        image_paths, figure_description = _collect_images(used_chunks)

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "citations": citations,
        "figure_description": figure_description,
        "retrieved_chunks": relevant_chunks,
        "image_path": image_paths[0] if image_paths else None,
        "image_paths": image_paths,
        "used_context": used_context,
    }


if __name__ == "__main__":
    for question in ["How do I view details of a node as a PCA?", "hii"]:
        result = generate_answer(question)
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Used context: {result['used_context']}")
        print(f"Sources: {result['sources']}")
        print(f"Citations: {result['citations']}")
        print(f"Images: {result['image_paths']}")
        print(f"Figure: {result['figure_description']}")
        print("-" * 40)






















































































































































# import json

# from app.rag import retrieve
# from app.llm_client import call_llm


# # this is called system prompt
# PROMPT_TEMPLATE = """You are a helpful assistant for the Singtel MEC Manager platform documentation. Answer the question using ONLY the context below as your source of truth.

# Context format
# Each block is labeled [Chunk N | Source: ...]. Some retrieved chunks may not be relevant to this question, even though they were retrieved. You must judge relevance yourself.

# How to answer
# - If the context describes relevant steps, screens, or information, even if not phrased exactly like the question, use it to give a clear, helpful answer.
# - If a procedure spans multiple chunks (e.g. steps 1-6 in one chunk, steps 7-12 in another), combine ALL of them into one complete answer, in order. Never answer with only the first part.
# - If the context is genuinely unrelated to the question, set used_context to false and give a brief, honest reply without inventing details.
# - Never invent steps, screens, fields, or details that are not in the context.

# Answer style (important)
# - Write in complete sentences, never a bare list of words. Start with a one-sentence direct answer to the question.
# - Then add the closely related details found in the SAME context that a user would need to act on the answer. Examples: what each value means, where in the UI to find it (menu path), the field or column name, caveats, warnings, prerequisites, limits, or what happens in edge cases.
# - Use only details present in the context. Do not pad with generic advice or repeat yourself.
# - For procedures, use a numbered list with one step per line. For lists of options or values, use bullets with a short explanation for each when the context provides one.
# - Use markdown (**bold** for UI labels and key terms). Inside the JSON string, write line breaks as \\n.
# - Typical length: 2-6 sentences for factual questions; as long as needed for procedures.

# Example of the expected style
# Question: What are the possible states of a transaction?
# Answer: "A transaction in the Transaction Logs can be in one of three states: **Complete**, **Failed**, or **In Progress**.\\n\\nThe state appears in the **State** column. While a transaction is still **In Progress**, its **Completed** timestamp is empty (null). To view the logs, go to **Services > Transaction Logs**."

# Figure descriptions
# Include figure_description whenever the answer is based on one or more figures. If multiple figures are used, join them in the ORDER THEY APPEAR IN THE DOCUMENT (by chunk/page order). This is about narrative flow, not relevance.

# Ranking chunks (controls which image is shown first)
# relevant_chunk_indices must be ordered from MOST relevant to LEAST relevant, regardless of where each chunk sits in the context. This is separate from figure_description's document-order rule. Only include a chunk index if you actually drew on that chunk's content.

# Context:
# {context}

# Question: {question}

# Respond with ONLY valid JSON in this exact shape, no other text, no markdown fences:
# {{"used_context": true or false, "answer": "your helpful, complete answer in markdown", "figure_description": "figure description(s) in document order, or empty string if none", "relevant_chunk_indices": [chunk numbers ordered from most to least relevant, e.g. [2, 0, 3]]}}"""



# def _chunk_text(c: dict) -> str:
#     """Text chunks store the passage under 'text'. Image chunks (from
#     ingest.py's build_page_image_points) don't have a 'text' key at all —
#     they store 'caption' and 'extracted_text' instead. Without this,
#     build_prompt() raises KeyError the moment an image chunk is retrieved
#     alongside text chunks, since both content types share one collection."""

#     if "text" in c:
#         return c["text"]
#     return " ".join(filter(None, [c.get("caption"), c.get("extracted_text")])) or "(no text)"


# # building the prompt using this function to formate and send to the llm for answer generation.
# def build_prompt(question: str, chunks: list[dict]) -> str:
#     """Combine retrieved chunks into a single, numbered context block and fill the prompt template.
#     Each chunk is tagged [Chunk N] so the LLM can report back exactly which
#     chunks it drew from, instead of judging the whole context as one lump.
#     """

#     context = "\n\n".join(
#         f"[Chunk {i} | Source: {c.get('source', 'unknown')}]\n{_chunk_text(c)}"
#         for i, c in enumerate(chunks)
#     )

#     return PROMPT_TEMPLATE.format(context=context, question=question)

# # parse the answer from the llm and return the answer
# def _parse_llm_json(raw: str) -> tuple[str, bool, list[int], str | None]:
#     """Parse the model's JSON response into (answer, used_context, relevant_chunk_indices, figure_description)."""
#     text = raw.strip()

#     if text.startswith("```"):
#         text = text.strip("`")
#         if text.startswith("json"):
#             text = text[4:]
#         text = text.strip()


#     start, end = text.find("{"), text.rfind("}")

#     if start != -1 and end != -1 and end > start:
#         text = text[start:end + 1]

#     try:
#         parsed = json.loads(text)
#         answer = parsed.get("answer", raw)
#         used_context = bool(parsed.get("used_context", False))
#         raw_indices = parsed.get("relevant_chunk_indices", [])
#         relevant_indices = [int(i) for i in raw_indices if isinstance(i, (int, float))]
#         figure_description = parsed.get("figure_description") or None
#         return answer, used_context, relevant_indices, figure_description
# # handle the error correctness here in this step    
#     except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
#         return raw, False, [], None





# def generate_answer(question: str) -> dict:
#     """Full RAG pipeline: retrieve relevant chunks, build prompt, call LLM.

#     retrieve() already drops low-score hits via score_threshold in rag.py,
#     so an empty chunks list means nothing relevant was found in the index.
#     On top of that, the LLM self-reports which specific chunks it actually
#     used (relevant_chunk_indices) — this catches the case where something
#     was retrieved and passed the score threshold, but wasn't actually
#     needed to answer this specific question. Only chunks confirmed as used
#     get their image/source attached to the response.
#     """

#     try:
#         chunks = retrieve(question)
#     except Exception as exc:
#         return {
#             "question": question,
#             "answer": "The documentation service is temporarily unavailable. Please make sure the vector search service is running, then try again.",
#             "sources": [],
#             "figure_description": None,
#             "retrieved_chunks": [],
#             "image_path": None,
#             "image_paths": [],
#             "used_context": False,
#             "error": f"Documentation search unavailable: {type(exc).__name__}",
#         }

#     if not chunks:

#         prompt = PROMPT_TEMPLATE.format(
#             context="No relevant documentation found.", question=question
#         )

#     else:
#         prompt = build_prompt(question, chunks)

#     try:
#         raw = call_llm(prompt)
#     except Exception as exc:
#         return {
#             "question": question,
#             "answer": "The AI answer service is temporarily unavailable. Please try again in a moment.",
#             "sources": [],
#             "figure_description": None,
#             "retrieved_chunks": [],
#             "image_path": None,
#             "image_paths": [],
#             "used_context": False,
#             "error": f"LLM service unavailable: {type(exc).__name__}",
#         }

#     answer, used_context, relevant_indices, figure_description = _parse_llm_json(raw)   # CHANGED: added figure_description

#     top_image = None
#     image_paths = []
#     sources = []
#     relevant_chunks = []
#     # REMOVED: figure_description = None   <-- no longer hardcoded here, it comes from the parser now

#     if used_context and chunks:
#         # only keep the chunks the model actually confirmed it used —
#         # not every chunk that got retrieved
#         used_chunks = [
#             chunks[i] for i in relevant_indices
#             if 0 <= i < len(chunks)
#         ]
#         # fallback: if the model said used_context=true but returned no/bad
#         # indices, don't silently drop everything — fall back to all retrieved
#         # chunks rather than showing nothing
#         if not used_chunks:
#             used_chunks = chunks

#         relevant_chunks = used_chunks
#         sources = [c.get("source", "unknown") for c in used_chunks]
#         image_chunks = [c for c in used_chunks if c.get("image_path")]
#         image_paths = [c["image_path"] for c in image_chunks]
#         top_image = image_paths[0] if image_paths else None
#     else:
#         # ADDED: don't trust a figure_description on an ungrounded answer
#         figure_description = None

#     return {
#         "question": question,
#         "answer": answer,
#         "sources": sources,
#         "figure_description": figure_description,
#         "retrieved_chunks": relevant_chunks,
#         "image_path": top_image,
#         "image_paths": image_paths,
#         "used_context": used_context,
#     }


# if __name__ == "__main__":
#     for question in ["How do I view notifications?", "hii"]:
#         result = generate_answer(question)
#         print(f"Question: {result['question']}")
#         print(f"Answer: {result['answer']}")
#         print(f"Used context: {result['used_context']}")
#         print(f"Sources: {result['sources']}")
#         print(f"Images: {result['image_paths']}")
#         print("-" * 40)















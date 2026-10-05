import json

from app.rag import retrieve
from app.llm_client import call_llm


# this is called system prompt
PROMPT_TEMPLATE = """You are a helpful assistant for the Singtel MEC Manager platform documentation. Answer the question using ONLY the context below as your source of truth.

Context format
Each block is labeled [Chunk N | Source: ...]. Some retrieved chunks may not be relevant to this question, even though they were retrieved. You must judge relevance yourself.

How to answer
- If the context describes relevant steps, screens, or information, even if not phrased exactly like the question, use it to give a clear, helpful answer.
- If a procedure spans multiple chunks (e.g. steps 1-6 in one chunk, steps 7-12 in another), combine ALL of them into one complete answer, in order. Never answer with only the first part.
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
Question: What are the possible states of a transaction?
Answer: "A transaction in the Transaction Logs can be in one of three states: **Complete**, **Failed**, or **In Progress**.\\n\\nThe state appears in the **State** column. While a transaction is still **In Progress**, its **Completed** timestamp is empty (null). To view the logs, go to **Services > Transaction Logs**."

Figure descriptions
Include figure_description whenever the answer is based on one or more figures. If multiple figures are used, join them in the ORDER THEY APPEAR IN THE DOCUMENT (by chunk/page order). This is about narrative flow, not relevance.

Ranking chunks (controls which image is shown first)
relevant_chunk_indices must be ordered from MOST relevant to LEAST relevant, regardless of where each chunk sits in the context. This is separate from figure_description's document-order rule. Only include a chunk index if you actually drew on that chunk's content.

Context:
{context}

Question: {question}

Respond with ONLY valid JSON in this exact shape, no other text, no markdown fences:
{{"used_context": true or false, "answer": "your helpful, complete answer in markdown", "figure_description": "figure description(s) in document order, or empty string if none", "relevant_chunk_indices": [chunk numbers ordered from most to least relevant, e.g. [2, 0, 3]]}}"""



def _chunk_text(c: dict) -> str:
    """Text chunks store the passage under 'text'. Image chunks (from
    ingest.py's build_page_image_points) don't have a 'text' key at all —
    they store 'caption' and 'extracted_text' instead. Without this,
    build_prompt() raises KeyError the moment an image chunk is retrieved
    alongside text chunks, since both content types share one collection."""

    if "text" in c:
        return c["text"]
    return " ".join(filter(None, [c.get("caption"), c.get("extracted_text")])) or "(no text)"


# building the prompt using this function to formate and send to the llm for answer generation.
def build_prompt(question: str, chunks: list[dict]) -> str:
    """Combine retrieved chunks into a single, numbered context block and fill the prompt template.
    Each chunk is tagged [Chunk N] so the LLM can report back exactly which
    chunks it drew from, instead of judging the whole context as one lump.
    """

    context = "\n\n".join(
        f"[Chunk {i} | Source: {c.get('source', 'unknown')}]\n{_chunk_text(c)}"
        for i, c in enumerate(chunks)
    )

    return PROMPT_TEMPLATE.format(context=context, question=question)

# parse the answer from the llm and return the answer
def _parse_llm_json(raw: str) -> tuple[str, bool, list[int], str | None]:
    """Parse the model's JSON response into (answer, used_context, relevant_chunk_indices, figure_description)."""
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
        relevant_indices = [int(i) for i in raw_indices if isinstance(i, (int, float))]
        figure_description = parsed.get("figure_description") or None
        return answer, used_context, relevant_indices, figure_description
# handle the error correctness here in this step    
    except (json.JSONDecodeError, TypeError, AttributeError, ValueError):
        return raw, False, [], None





def generate_answer(question: str) -> dict:
    """Full RAG pipeline: retrieve relevant chunks, build prompt, call LLM.

    retrieve() already drops low-score hits via score_threshold in rag.py,
    so an empty chunks list means nothing relevant was found in the index.
    On top of that, the LLM self-reports which specific chunks it actually
    used (relevant_chunk_indices) — this catches the case where something
    was retrieved and passed the score threshold, but wasn't actually
    needed to answer this specific question. Only chunks confirmed as used
    get their image/source attached to the response.
    """

    try:
        chunks = retrieve(question)
    except Exception as exc:
        return {
            "question": question,
            "answer": "The documentation service is temporarily unavailable. Please make sure the vector search service is running, then try again.",
            "sources": [],
            "figure_description": None,
            "retrieved_chunks": [],
            "image_path": None,
            "image_paths": [],
            "used_context": False,
            "error": f"Documentation search unavailable: {type(exc).__name__}",
        }

    if not chunks:

        prompt = PROMPT_TEMPLATE.format(
            context="No relevant documentation found.", question=question
        )

    else:
        prompt = build_prompt(question, chunks)

    try:
        raw = call_llm(prompt)
    except Exception as exc:
        return {
            "question": question,
            "answer": "The AI answer service is temporarily unavailable. Please try again in a moment.",
            "sources": [],
            "figure_description": None,
            "retrieved_chunks": [],
            "image_path": None,
            "image_paths": [],
            "used_context": False,
            "error": f"LLM service unavailable: {type(exc).__name__}",
        }

    answer, used_context, relevant_indices, figure_description = _parse_llm_json(raw)   # CHANGED: added figure_description

    top_image = None
    image_paths = []
    sources = []
    relevant_chunks = []
    # REMOVED: figure_description = None   <-- no longer hardcoded here, it comes from the parser now

    if used_context and chunks:
        # only keep the chunks the model actually confirmed it used —
        # not every chunk that got retrieved
        used_chunks = [
            chunks[i] for i in relevant_indices
            if 0 <= i < len(chunks)
        ]
        # fallback: if the model said used_context=true but returned no/bad
        # indices, don't silently drop everything — fall back to all retrieved
        # chunks rather than showing nothing
        if not used_chunks:
            used_chunks = chunks

        relevant_chunks = used_chunks
        sources = [c.get("source", "unknown") for c in used_chunks]
        image_chunks = [c for c in used_chunks if c.get("image_path")]
        image_paths = [c["image_path"] for c in image_chunks]
        top_image = image_paths[0] if image_paths else None
    else:
        # ADDED: don't trust a figure_description on an ungrounded answer
        figure_description = None

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "figure_description": figure_description,
        "retrieved_chunks": relevant_chunks,
        "image_path": top_image,
        "image_paths": image_paths,
        "used_context": used_context,
    }


if __name__ == "__main__":
    for question in ["How do I view notifications?", "hii"]:
        result = generate_answer(question)
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print(f"Used context: {result['used_context']}")
        print(f"Sources: {result['sources']}")
        print(f"Images: {result['image_paths']}")
        print("-" * 40)















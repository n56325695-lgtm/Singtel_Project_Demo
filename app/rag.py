"""
retrieve.py  --  search Qdrant for the chunks that answer a question.
Same shape as your old retrieve(), plus:
  * persona filter  (PCA / MCA / CTP) - explicit argument, or auto-detected from the query
    only when the query names exactly ONE role. If the filtered search finds nothing it
    falls back to an unfiltered search, so a wrong guess can never hide the answer.
  * every result carries `image_paths` (all screenshots of that chunk) and `section_path`
    (use it for citations); `image_path` (first image) is kept for backward compatibility.
  * text_chunk and table_chunk are both plain `text`; there is no separate image type now.
"""
import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


from app.config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    EMBEDDING_MODEL, TOP_K, SIMILARITY_THRESHOLD,
)

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

# words in a question that name a role -> the persona value stored by extract.py
PERSONA_KEYWORDS = {
    "Private Cloud Administrator": ("private cloud administrator", "pca"),
    "Multi Cloud Administrator": ("multi cloud administrator", "multi-cloud administrator", "mca"),
    "Cloud Tenant Portal Administrator": ("cloud tenant portal", "tenant portal", "ctp"),
}



def embed_query(query: str) -> list[float]:
    """Same Ollama model as ingestion."""
    return ollama.embed(model=EMBEDDING_MODEL, input=query)["embeddings"][0]


def detect_persona(query: str) -> str | None:
    q = f" {query.lower()} "
    hits = [p for p, words in PERSONA_KEYWORDS.items()
            if any(f" {w} " in q or f" {w}?" in q or f" {w}," in q or f" {w}." in q for w in words)]
    return hits[0] if len(hits) == 1 else None      # ambiguous or none -> no filter


def _search(query_vector, top_k, min_score, persona):
    flt = None
    if persona:
        flt = Filter(must=[FieldCondition(key="persona", match=MatchValue(value=persona))])
    try:  # qdrant-client >= 1.10
        return client.query_points(collection_name=COLLECTION_NAME, query=query_vector, limit=top_k,
                                   score_threshold=min_score, query_filter=flt,
                                   with_payload=True).points
    except AttributeError:  # older client
        return client.search(collection_name=COLLECTION_NAME, query_vector=query_vector, limit=top_k,
                             score_threshold=min_score, query_filter=flt)


def retrieve(query: str, top_k: int = TOP_K, min_score: float = SIMILARITY_THRESHOLD,
             persona: str | None = None) -> list[dict]:
    """Embed the query and return the most similar chunks (empty list if nothing
    clears min_score, e.g. greetings / off-topic questions)."""
    query_vector = embed_query(query)
    persona = persona or detect_persona(query)

    results = _search(query_vector, top_k, min_score, persona)
    if persona and not results:                      # wrong guess must never hide the answer
        results = _search(query_vector, top_k, min_score, None)
        persona = None

    chunks = []
    for hit in results:
        p = hit.payload
        chunks.append({
            "text": p.get("text", ""),
            "source": p.get("source"),
            "content_type": p.get("content_type"),
            "persona": p.get("persona"),
            "section_path": p.get("section_path"),
            "pages": p.get("pages"),
            "page_num": p.get("page_num"),
            "image_paths": p.get("image_paths") or [],
            "image_path": p.get("image_path"),
            "score": hit.score,
            "id": hit.id,
        })
    return chunks


if __name__ == "__main__":
    for query in ["How do I view node details as a PCA?", "What is a Node Size?", "hii"]:
        results = retrieve(query)
        print(f"Query: {query}")
        if not results:
            print(f"  No results above threshold ({SIMILARITY_THRESHOLD})\n")
            continue
        for r in results:
            print(f"  [{r['score']:.3f}] {r['section_path']} (p{r['page_num']}) "
                  f"imgs={len(r['image_paths'])} {r['text'][:80]!r}")
        print()


















































# import ollama
# from qdrant_client import QdrantClient

# from app.config import (
#     QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
#     EMBEDDING_MODEL, TOP_K, SIMILARITY_THRESHOLD
# )

# client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


# def embed_query(query: str) -> list[float]:
#     """Generate query embedding using the same Ollama model used during ingestion."""
#     response = ollama.embed(
#         model=EMBEDDING_MODEL,
#         input=query
#     )
#     return response["embeddings"][0]



# # this is the step where i retrived the chunks from the qdrant vector db 
# def retrieve(query: str, top_k: int = TOP_K, min_score: float = SIMILARITY_THRESHOLD) -> list[dict]:
#     """Embed the query and search Qdrant for the most similar chunks.
#     Handles both content types: text_chunk (has 'text') and page_image
#     (has 'extracted_text' + 'caption', combined here into 'text').

#     score_threshold drops hits below min_score so unrelated queries
#     (greetings, small talk, off-topic questions) come back as an empty
#     list instead of the nearest-but-irrelevant neighbors. Qdrant applies
#     this filter server-side before results ever reach this function."""
#     query_vector = embed_query(query)

#     results = client.search(
#         collection_name=COLLECTION_NAME,
#         query_vector=query_vector,
#         limit=top_k,
#         score_threshold=min_score,
#     )
    
#     chunks = []
#     for hit in results:
#         payload = hit.payload
# # for handle the content type image
#         if payload.get("content_type") == "page_image":
#             text = " ".join(filter(None, [payload.get("caption"), payload.get("extracted_text")]))
#         else:
# # for handle the content type text_chunk
#             text = payload.get("text", "")

# # //this is the formate need to use to save the chunks in the vector db 
#         chunks.append({
#             "text": text,
#             "source": payload.get("source"),
#             "content_type": payload.get("content_type"),
#             "image_path": payload.get("image_path"),
#             "page_num": payload.get("page_num"),
#             "score": hit.score,
#             "id": hit.id,
#         })

#     return chunks


# if __name__ == "__main__":
#     for query in ["what is singtel?", "hii"]:
#         results = retrieve(query)
#         print(f"Query: {query}")
#         if not results:
#             print(f"  No results above threshold ({SIMILARITY_THRESHOLD})\n")
#             continue
#         for r in results:
#             print(f"  [{r['score']:.3f}] ({r['source']}) img={r['image_path']} {r['text'][:100]}...")
#         print()






























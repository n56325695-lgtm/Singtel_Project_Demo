import ollama
from qdrant_client import QdrantClient

from app.config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    EMBEDDING_MODEL, TOP_K, SIMILARITY_THRESHOLD
)

client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def embed_query(query: str) -> list[float]:
    """Generate query embedding using the same Ollama model used during ingestion."""
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )
    return response["embeddings"][0]


# this is the step where i retrived the chunks from the qdrant vector db 
def retrieve(query: str, top_k: int = TOP_K, min_score: float = SIMILARITY_THRESHOLD) -> list[dict]:
    """Embed the query and search Qdrant for the most similar chunks.
    Handles both content types: text_chunk (has 'text') and page_image
    (has 'extracted_text' + 'caption', combined here into 'text').

    score_threshold drops hits below min_score so unrelated queries
    (greetings, small talk, off-topic questions) come back as an empty
    list instead of the nearest-but-irrelevant neighbors. Qdrant applies
    this filter server-side before results ever reach this function."""
    query_vector = embed_query(query)

    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        score_threshold=min_score,
    )
    
    chunks = []
    for hit in results:
        payload = hit.payload
# for handle the content type image
        if payload.get("content_type") == "page_image":
            text = " ".join(filter(None, [payload.get("caption"), payload.get("extracted_text")]))
        else:
# for handle the content type text_chunk
            text = payload.get("text", "")

# //this is the formate need to use to save the chunks in the vector db 
        chunks.append({
            "text": text,
            "source": payload.get("source"),
            "content_type": payload.get("content_type"),
            "image_path": payload.get("image_path"),
            "page_num": payload.get("page_num"),
            "score": hit.score,
            "id": hit.id,
        })

    return chunks


if __name__ == "__main__":
    for query in ["what is singtel?", "hii"]:
        results = retrieve(query)
        print(f"Query: {query}")
        if not results:
            print(f"  No results above threshold ({SIMILARITY_THRESHOLD})\n")
            continue
        for r in results:
            print(f"  [{r['score']:.3f}] ({r['source']}) img={r['image_path']} {r['text'][:100]}...")
        print()






























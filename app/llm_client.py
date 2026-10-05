import requests
from groq import Groq
from app.config import (
    EMBEDDING_MODEL,
    OLLAMA_HOST,
    GROQ_API_KEY,
    GROQ_MODEL,
)
# Initialize the official Groq SDK client
groq_client = Groq(api_key=GROQ_API_KEY)

def _call_groq(prompt: str) -> str:
    """Send a prompt to Groq using the official SDK to bypass Cloudflare blocks."""
    try:
        chat_completion = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_completion_tokens=2048,
            top_p=1,
            stream=False,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(f"\n[GROQ API ERROR] Details: {e}\n")
        raise e

# call the llm using the groq sdk 
def call_llm(prompt: str) -> str:
    """Using Groq SDK for chat completions."""
    return _call_groq(prompt)


def embed_text(text: str) -> list[float]:
    """bge-m3 embedding — ALWAYS via Ollama. 
    Do not comment this out."""
    resp = requests.post(
        f"{OLLAMA_HOST}/api/embeddings",
        json={"model": EMBEDDING_MODEL, "prompt": text},
    )
    resp.raise_for_status()
    return resp.json()["embedding"]













































# import json
# import requests
# from urllib.error import HTTPError
# from urllib.request import Request, urlopen

# from app.config import (
#     EMBEDDING_MODEL,
#     OLLAMA_HOST,
#     GROQ_API_KEY,
#     GROQ_MODEL,
# )


# # --- Ollama LLM path: commented out, using Groq for now ---
# # def _call_ollama(prompt: str) -> str:
# #     """Send a prompt to a local Ollama model and return the text response."""
# #     response = requests.post(
# #         f"{OLLAMA_HOST}/api/generate",
# #         json={
# #             "model": LLM_MODEL,
# #             "prompt": prompt,
# #             "stream": False,
# #             "options": {"temperature": 0.2},
# #         },
# #     )
# #     response.raise_for_status()
# #     return response.json()["response"]


# def _call_groq(prompt: str) -> str:
#     """Send a prompt to Groq's hosted API and return the text response."""
#     request = Request(
#         "https://api.groq.com/openai/v1/chat/completions",
#         data=json.dumps(
#             {
#                 "model": GROQ_MODEL,
#                 "messages": [{"role": "user", "content": prompt}],
#                 "temperature": 0.2,
#                 "max_completion_tokens": 2048,
#                 "top_p": 1,
#                 "stream": False,
#             }
#         ).encode("utf-8"),
#         headers={
#             "Authorization": f"Bearer {GROQ_API_KEY}",
#             "Content-Type": "application/json",
#         },
#         method="POST",
#     )
#     try:
#         with urlopen(request) as response:
#             completion = json.load(response)
#         return completion["choices"][0]["message"]["content"]
#     except HTTPError as e:
#         # Read the exact error message sent by Groq's servers
#         error_details = e.read().decode("utf-8")
#         print(f"\n[GROQ API ERROR {e.code}] Response details: {error_details}\n")
#         raise e
# def call_llm(prompt: str) -> str:
#     """Using Groq only for now. Ollama LLM path is commented out above."""
#     return _call_groq(prompt)


# def embed_text(text: str) -> list[float]:
#     """bge-m3 embedding — ALWAYS via Ollama, regardless of which LLM
#     provider is active above. Groq does not host embedding models, so
#     this call is required even while using Groq for chat completions.
#     Do not comment this out."""
#     resp = requests.post(
#         f"{OLLAMA_HOST}/api/embeddings",
#         json={"model": EMBEDDING_MODEL, "prompt": text},
#     )
#     resp.raise_for_status()
#     return resp.json()["embedding"]





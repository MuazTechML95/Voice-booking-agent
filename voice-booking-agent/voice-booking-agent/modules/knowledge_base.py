"""
modules/knowledge_base.py
----------------------------
Retrieval-Augmented Generation (RAG) module for answering business
FAQs, services, and pricing questions during the conversation.

Design choices (per project requirement to stay OpenAI-only, no local
vector-DB library):
    - Embeddings: OpenAI `text-embedding-3-small` via the OpenAI API.
    - Vector "store": a small local JSON cache (database/embeddings_cache.json)
      holding chunk text + embedding vectors. Similarity search is plain
      cosine similarity computed with numpy — no Chroma/FAISS/etc.
    - Generation: OpenAI chat completion, grounded strictly on the
      retrieved chunks (the prompt instructs the model not to use
      outside knowledge).

Content is fully config-driven: each business type points to a plain
text file under config/knowledge/ (see config.json -> "knowledge_files").
Editing those .txt files is enough to update a business's FAQs/services/
pricing — no code changes required, keeping the project generic.

If no OPENAI_API_KEY is configured, or any API call fails, this module
fails safely: it returns (False, friendly_message) instead of raising,
so the booking flow is never interrupted.
"""

import hashlib
import json
import os

import numpy as np

from modules.utils import BASE_DIR, get_logger

logger = get_logger(__name__)

KNOWLEDGE_DIR = os.path.join(BASE_DIR, "config", "knowledge")
CACHE_PATH = os.path.join(BASE_DIR, "database", "embeddings_cache.json")

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
TOP_K = 3
MIN_SIMILARITY = 0.15  # below this, we treat retrieval as "no relevant info found"

# Used to decide whether a user message should be routed to RAG instead
# of the normal booking flow. Kept simple/rule-based so routing works
# even when the OpenAI API itself is unavailable.
QUESTION_STARTERS = (
    "what", "how", "when", "where", "why", "which", "who",
    "do you", "does your", "are you", "is there", "can i", "could i",
    "kya", "kab", "kahan", "kitna", "kitni", "kaise",
)
INFO_KEYWORDS = (
    "hour", "open", "close", "timing", "price", "cost", "fee", "charge",
    "service", "address", "location", "policy", "discount", "package",
    "insurance", "warranty", "scholarship", "parking",
)


def looks_like_knowledge_query(text: str, strict: bool = False) -> bool:
    """
    Heuristic check: does this message look like an info/FAQ question
    rather than a direct answer to a booking-flow prompt?

    `strict=True` is used while we're in the middle of collecting a
    specific field (name/phone/date/time/purpose) to avoid accidentally
    hijacking a legitimate answer that happens to contain a keyword
    (e.g. "general checkup" as a purpose). In strict mode we only trigger
    on an explicit question mark or a clear question-starting phrase.
    """
    t = (text or "").strip().lower()
    if not t:
        return False
    if t.endswith("?"):
        return True
    if any(t.startswith(w) for w in QUESTION_STARTERS):
        return True
    if not strict and any(k in t for k in INFO_KEYWORDS):
        return True
    return False


# --------------------------------------------------------------------------- #
# OpenAI client helper
# --------------------------------------------------------------------------- #
def _get_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        return OpenAI(api_key=api_key)
    except ImportError:
        logger.warning("openai package not installed; RAG answers are unavailable.")
        return None


# --------------------------------------------------------------------------- #
# Knowledge source loading & chunking
# --------------------------------------------------------------------------- #
def _slugify(business_type: str) -> str:
    return business_type.strip().lower().replace(" ", "_")


def _load_source_text(config: dict, business_type: str) -> str:
    files_map = config.get("knowledge_files", {})
    filename = files_map.get(business_type) or files_map.get("default")
    if not filename:
        return ""
    path = os.path.join(KNOWLEDGE_DIR, filename)
    if not os.path.exists(path):
        logger.warning("Knowledge file not found: %s", path)
        return ""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _chunk_text(text: str, max_chars: int = 500) -> list:
    """Group the knowledge file into paragraph-sized chunks for embedding."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks, buf = [], ""
    for p in paragraphs:
        if buf and len(buf) + len(p) > max_chars:
            chunks.append(buf)
            buf = p
        else:
            buf = f"{buf}\n\n{p}".strip()
    if buf:
        chunks.append(buf)
    return chunks


# --------------------------------------------------------------------------- #
# Local embedding cache (JSON file, not a vector-DB library)
# --------------------------------------------------------------------------- #
def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_cache(cache: dict):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(cache, f)


def _embed_texts(client, texts: list) -> list:
    response = client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


def _get_or_build_index(config: dict, business_type: str):
    """
    Return (client, chunks, embeddings_matrix) for a business type,
    rebuilding the embedding cache only if the source .txt file changed
    (tracked via a content hash) so we don't re-call the embeddings API
    on every single question.
    """
    client = _get_client()
    if client is None:
        return None, None, None

    source_text = _load_source_text(config, business_type)
    if not source_text.strip():
        return None, None, None

    source_hash = hashlib.sha256(source_text.encode("utf-8")).hexdigest()
    cache = _load_cache()
    key = _slugify(business_type)
    cached = cache.get(key)

    if cached and cached.get("hash") == source_hash:
        return client, cached["chunks"], np.array(cached["embeddings"], dtype=float)

    chunks = _chunk_text(source_text)
    if not chunks:
        return None, None, None

    try:
        embeddings_list = _embed_texts(client, chunks)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Embedding generation failed: %s", exc)
        return None, None, None

    cache[key] = {"hash": source_hash, "chunks": chunks, "embeddings": embeddings_list}
    _save_cache(cache)
    return client, chunks, np.array(embeddings_list, dtype=float)


def _cosine_similarity(query_vec, matrix: np.ndarray) -> np.ndarray:
    query_vec = np.array(query_vec, dtype=float)
    norm_q = np.linalg.norm(query_vec)
    norm_m = np.linalg.norm(matrix, axis=1)
    denom = norm_m * norm_q
    denom[denom == 0] = 1e-10
    return (matrix @ query_vec) / denom


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def retrieve(config: dict, business_type: str, query: str, top_k: int = TOP_K) -> list:
    """Return the most relevant knowledge chunks for a query, or [] if unavailable."""
    client, chunks, embeddings = _get_or_build_index(config, business_type)
    if client is None or not chunks:
        return []

    try:
        query_embedding = _embed_texts(client, [query])[0]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Query embedding failed: %s", exc)
        return []

    sims = _cosine_similarity(query_embedding, embeddings)
    ranked = np.argsort(sims)[::-1][:top_k]
    return [chunks[i] for i in ranked if sims[i] >= MIN_SIMILARITY]


def answer_question(config: dict, business_type: str, query: str):
    """
    Full RAG pipeline for one user question.
    Returns (ok: bool, answer: str). `ok=False` means a graceful
    fallback message is returned instead of a generated answer.
    """
    client = _get_client()
    if client is None:
        return False, (
            "I can't look that up right now (no OpenAI API key configured). "
            "Please contact us directly for that, or let's continue with your booking."
        )

    context_chunks = retrieve(config, business_type, query)
    if not context_chunks:
        return False, (
            "I don't have specific information about that. "
            "Please contact us directly, or let's continue with your booking."
        )

    context = "\n\n---\n\n".join(context_chunks)
    business_name = config.get("business_name", "the business")

    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"You are a helpful assistant for {business_name} ({business_type}). "
                        "Answer the user's question using ONLY the context below. "
                        "If the answer is not in the context, say you don't have that "
                        "information and suggest contacting the business directly. "
                        "Keep the answer short and friendly (2-3 sentences).\n\n"
                        f"Context:\n{context}"
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=200,
            temperature=0.2,
        )
        return True, response.choices[0].message.content.strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning("RAG answer generation failed: %s", exc)
        return False, "Sorry, I couldn't process that question right now. Please try again."

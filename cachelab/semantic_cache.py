"""Semantic cache: find a stored question that MEANS the same thing, using Redis vector search.

Each entry is a Redis hash, like the exact cache, plus the question's embedding:

    cache:semantic:<uuid>
        question, answer, model, output_tokens, created_at, hits
        embedding   768 float32 numbers packed as raw bytes (3 KB)

A vector index (FT.CREATE) over those hashes lets Redis answer
"which stored embedding is closest to this one?" (KNN search) in milliseconds.
"""

import re
import time
import uuid

import numpy as np
import redis

from cachelab.embeddings import DIM, embed
from cachelab.redis_client import get_redis

PREFIX = "cache:semantic:"
INDEX = "idx:semantic_cache"
DEFAULT_THRESHOLD = 0.82  # measured for nomic-embed-text: paraphrases scored 0.84+, "reset my router" 0.79
DEFAULT_TTL = 3600


_index_ready = False


def ensure_index() -> None:
    """Create the vector index once. Redis then indexes every hash whose key starts with PREFIX."""
    global _index_ready
    if _index_ready:  # skip the FT.INFO round trip on every request
        return
    r = get_redis()
    try:
        r.execute_command("FT.INFO", INDEX)
        _index_ready = True
        return
    except redis.ResponseError:
        pass  # "Unknown index name" -> create it
    r.execute_command(
        "FT.CREATE", INDEX, "ON", "HASH", "PREFIX", "1", PREFIX,
        "SCHEMA",
        "model", "TAG",                     # filter: only match answers from the same model
        "question", "TEXT",
        "embedding", "VECTOR", "HNSW", "6",  # HNSW = fast approximate nearest-neighbour graph
            "TYPE", "FLOAT32", "DIM", str(DIM), "DISTANCE_METRIC", "COSINE",
    )
    _index_ready = True


def _tag(value: str) -> str:
    # TAG queries treat punctuation as syntax, so "llama3.2:3b" must be sent as "llama3\.2\:3b".
    return re.sub(r"([^A-Za-z0-9_])", r"\\\1", value)


def nearest(vec: np.ndarray, model: str) -> dict | None:
    """The single closest stored question for this model, whatever its similarity."""
    r = get_redis()
    ensure_index()
    res = r.execute_command(
        "FT.SEARCH", INDEX,
        f"(@model:{{{_tag(model)}}})=>[KNN 1 @embedding $vec AS distance]",
        "PARAMS", "2", "vec", vec.astype(np.float32).tobytes(),
        "SORTBY", "distance",
        "RETURN", "4", "question", "answer", "output_tokens", "distance",
        "DIALECT", "2",
    )
    # redis-py 8 speaks RESP3 by default, so the reply is a map:
    #   {"total_results": 1, "results": [{"id": key, "extra_attributes": {field: value, ...}}]}
    # (Older RESP2 clients get a flat list instead: [total, key1, [field, value, ...], ...])
    if isinstance(res, dict):
        if not res.get("results"):
            return None
        key, doc = res["results"][0]["id"], res["results"][0]["extra_attributes"]
    else:
        if not res or res[0] == 0:
            return None
        key, fields = res[1], res[2]
        doc = dict(zip(fields[::2], fields[1::2]))
    return {
        "key": key,
        "question": doc["question"],
        "answer": doc["answer"],
        "output_tokens": int(doc.get("output_tokens", 0)),
        "similarity": 1 - float(doc["distance"]),  # Redis returns cosine DISTANCE = 1 - similarity
    }


def lookup(question: str, model: str, threshold: float = DEFAULT_THRESHOLD) -> dict:
    """Embed + KNN search. Always reports the best match found, and whether it clears the threshold."""
    vec, embed_ms = embed(question)
    start = time.perf_counter()
    match = nearest(vec, model)
    search_ms = (time.perf_counter() - start) * 1000

    hit = match is not None and match["similarity"] >= threshold
    if hit:
        get_redis().hincrby(match["key"], "hits", 1)
    return {"hit": hit, "match": match, "vector": vec, "embed_ms": embed_ms, "search_ms": search_ms}


def store(question: str, model: str, answer: str, output_tokens: int,
          vec: np.ndarray, ttl: int = DEFAULT_TTL) -> str:
    """Save a new entry. Pass the vector from lookup() so we don't embed the same text twice."""
    r = get_redis()
    ensure_index()
    key = f"{PREFIX}{uuid.uuid4().hex}"
    pipe = r.pipeline()
    pipe.hset(key, mapping={
        "question": question,
        "answer": answer,
        "model": model,
        "output_tokens": output_tokens,
        "created_at": time.time(),
        "hits": 0,
        "embedding": vec.astype(np.float32).tobytes(),
    })
    pipe.expire(key, ttl)  # expired hashes drop out of the index automatically
    pipe.execute()
    return key


def entries() -> list[dict]:
    r = get_redis()
    rows = []
    for key in r.scan_iter(match=f"{PREFIX}*", count=100):
        # HMGET named fields only: the embedding is raw bytes and isn't valid text.
        q, hits, tokens = r.hmget(key, "question", "hits", "output_tokens")
        if q is not None:
            rows.append({"key": key, "question": q, "hits": int(hits or 0),
                         "output_tokens": int(tokens or 0), "ttl_left_s": r.ttl(key)})
    return rows


def clear() -> int:
    r = get_redis()
    keys = list(r.scan_iter(match=f"{PREFIX}*", count=100))
    return r.delete(*keys) if keys else 0

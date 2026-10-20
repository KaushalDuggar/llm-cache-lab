"""Exact-match cache: same (normalized) text -> same Redis key -> stored answer.

Each entry is a Redis hash:

    cache:exact:<model>:<sha256 of normalized question>
        question       original wording that created the entry
        answer         the LLM's answer
        output_tokens  how many tokens the LLM spent generating it
        created_at     unix time
        hits           how many times this entry has been reused

and expires after `ttl` seconds (EXPIRE), so stale answers don't live forever.
"""

import hashlib
import re
import time

from cachelab.redis_client import get_redis

PREFIX = "cache:exact"
DEFAULT_TTL = 3600  # 1 hour


def normalize(question: str) -> str:
    """Make trivially different text identical: case, extra spaces, trailing punctuation.

    "  How do I reset my PASSWORD?? " -> "how do i reset my password"
    Paraphrases ("I forgot my login password") still differ. That's semantic caching's job.
    """
    text = question.strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text.rstrip("?!. ")


def make_key(question: str, model: str) -> str:
    # The model is part of the key: llama's answer shouldn't be served for a qwen request.
    digest = hashlib.sha256(normalize(question).encode()).hexdigest()
    return f"{PREFIX}:{model}:{digest}"


def lookup(question: str, model: str) -> dict | None:
    r = get_redis()
    key = make_key(question, model)
    # Every Redis call is a network round trip (~25 ms to the cloud), so batch the reads into one.
    entry, ttl_left = r.pipeline(transaction=False).hgetall(key).ttl(key).execute()
    if not entry:  # {} when the key doesn't exist (or has expired)
        return None
    r.hincrby(key, "hits", 1)
    return {
        "key": key,
        "question": entry["question"],
        "answer": entry["answer"],
        "output_tokens": int(entry.get("output_tokens", 0)),
        "created_at": float(entry.get("created_at", 0)),
        "ttl_left": ttl_left,
    }


def store(question: str, model: str, answer: str, output_tokens: int, ttl: int = DEFAULT_TTL) -> str:
    r = get_redis()
    key = make_key(question, model)
    # MULTI/EXEC so the entry and its expiry are written together.
    pipe = r.pipeline()
    pipe.hset(key, mapping={
        "question": question,
        "answer": answer,
        "output_tokens": output_tokens,
        "created_at": time.time(),
        "hits": 0,
    })
    pipe.expire(key, ttl)
    pipe.execute()
    return key


def entries() -> list[dict]:
    """All live entries. SCAN walks the keyspace in batches instead of blocking Redis like KEYS."""
    r = get_redis()
    rows = []
    for key in r.scan_iter(match=f"{PREFIX}:*", count=100):
        e = r.hgetall(key)
        if e:
            rows.append({"key": key, "question": e.get("question"), "hits": int(e.get("hits", 0)),
                         "output_tokens": int(e.get("output_tokens", 0)), "ttl_left_s": r.ttl(key)})
    return rows


def clear() -> int:
    r = get_redis()
    keys = list(r.scan_iter(match=f"{PREFIX}:*", count=100))
    return r.delete(*keys) if keys else 0

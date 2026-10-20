"""The request "router": decides whether an answer comes from a cache or the LLM.

    cache_mode="none"      -> always LLM (baseline)
    cache_mode="exact"     -> Redis exact-match lookup first; LLM only on a miss
    cache_mode="semantic"  -> embed the question, Redis vector search for a similar one
    cache_mode="hybrid"    -> exact first (cheapest), then semantic, then LLM
"""

import time

from cachelab import exact_cache, llm, metrics, semantic_cache
from cachelab.redis_client import incr_stat

CACHE_MODES = ["none", "exact", "semantic", "hybrid"]


def _log(hit: bool, source: str, latency_ms: float, tokens: int, question: str,
         similarity: float | None = None) -> None:
    # Printed to the terminal running Streamlit, so you can watch the routing live.
    tag = "[HIT] " if hit else "[MISS]"
    sim = f"sim {similarity:.3f}" if similarity is not None else " " * 9
    print(f"{tag} {source:<14} {latency_ms:>7.0f}ms {tokens:>5} tok  {sim}  {question[:60]!r}", flush=True)


def ask(
    question: str,
    model: str = llm.DEFAULT_CHAT_MODEL,
    cache_mode: str = "none",
    ttl: int = exact_cache.DEFAULT_TTL,
    threshold: float = semantic_cache.DEFAULT_THRESHOLD,
) -> dict:
    question = question.strip()
    start = time.perf_counter()
    uses_cache = cache_mode != "none"
    if uses_cache:
        incr_stat("requests")

    def hit(source: str, answer: str, tokens: int, matched: str, key: str, **extra) -> dict:
        latency_ms = (time.perf_counter() - start) * 1000
        incr_stat("hits")
        incr_stat(f"hits_{source}")
        metrics.record(
            question=question, answer=answer, model=model, cache_mode=cache_mode,
            source=source, latency_ms=latency_ms,
            output_tokens=tokens,  # tokens we didn't have to generate
            similarity=extra.get("similarity"), matched_question=matched,
        )
        _log(True, source, latency_ms, tokens, question, extra.get("similarity"))
        return {
            "question": question, "answer": answer, "source": source, "cache_hit": True,
            "latency_ms": latency_ms, "load_ms": 0.0, "output_tokens": tokens, "tokens_per_sec": 0.0,
            "cache_key": key, "matched_question": matched, **extra,
        }

    # --- route 1: exact match (one hash lookup, no embedding needed) ---
    if cache_mode in ("exact", "hybrid"):
        cached = exact_cache.lookup(question, model)
        if cached:
            return hit("exact_cache", cached["answer"], cached["output_tokens"], cached["question"],
                       cached["key"], ttl_left=cached["ttl_left"])

    # --- route 2: semantic match (embed -> KNN vector search -> threshold check) ---
    sem = None
    if cache_mode in ("semantic", "hybrid"):
        sem = semantic_cache.lookup(question, model, threshold)
        if sem["hit"]:
            m = sem["match"]
            return hit("semantic_cache", m["answer"], m["output_tokens"], m["question"], m["key"],
                       similarity=m["similarity"], threshold=threshold,
                       embed_ms=sem["embed_ms"], search_ms=sem["search_ms"])

    # --- route 3: MISS, ask the LLM, then remember the answer in every cache in play ---
    result = llm.chat(question, model)
    cache_key = None
    if cache_mode in ("exact", "hybrid"):
        cache_key = exact_cache.store(question, model, result.text, result.output_tokens, ttl)
    if sem is not None:
        cache_key = semantic_cache.store(question, model, result.text, result.output_tokens,
                                         sem["vector"], ttl)
    if uses_cache:
        incr_stat("misses")
    latency_ms = (time.perf_counter() - start) * 1000  # includes the failed lookups + the stores

    # On a semantic miss, still log how close the nearest entry was: that's how you tune the threshold.
    nearest = sem["match"] if sem else None
    similarity = nearest["similarity"] if nearest else None
    metrics.record(
        question=question, answer=result.text, model=model, cache_mode=cache_mode,
        source="llm", latency_ms=latency_ms, load_ms=result.load_ms,
        output_tokens=result.output_tokens,
        similarity=similarity, matched_question=nearest["question"] if nearest else None,
    )
    _log(False, "llm", latency_ms, result.output_tokens, question, similarity)
    return {
        "question": question, "answer": result.text, "source": "llm", "cache_hit": False,
        "latency_ms": latency_ms, "load_ms": result.load_ms,
        "output_tokens": result.output_tokens, "tokens_per_sec": result.tokens_per_sec,
        "cache_key": cache_key, "matched_question": None, "ttl_left": ttl if cache_key else None,
        "similarity": similarity, "nearest_question": nearest["question"] if nearest else None,
        "threshold": threshold if sem else None,
        "embed_ms": sem["embed_ms"] if sem else None, "search_ms": sem["search_ms"] if sem else None,
    }

# LLM Cache Lab

A hands-on lab for learning how **exact-match** and **semantic** caching speed up LLM apps.
Everything runs locally with [Ollama](https://ollama.com), with Redis as the cache and a
Streamlit dashboard that shows hit rate, miss rate, latency, and tokens saved as you go.

```
question ──► exact cache (Redis hash) ──hit──► answer   ~90 ms
                 │ miss
                 ▼
            semantic cache (Redis vector search) ──hit──► answer   ~145 ms
                 │ miss
                 ▼
            LLM (Ollama) ──► store in both caches ──► answer   ~9 s
```

## What it shows

| Cache | Matches | How |
|---|---|---|
| **Exact** | Identical text after normalizing (case, spaces, trailing `?`) | `sha256` → `HGETALL cache:exact:<model>:<hash>` |
| **Semantic** | Same *meaning*, different words | Ollama embedding → Redis `FT.SEARCH` KNN → cosine similarity ≥ threshold |
| **Hybrid** | Both, cheapest first | exact → semantic → LLM |

Sample run (10 questions, hybrid, threshold 0.82, `llama3.2:3b` on an M-series Mac, Redis Cloud):

| | Result |
|---|---|
| Hit rate | 60% (2 exact, 4 semantic) |
| Avg miss / avg hit | 8,745 ms / 126 ms |
| Tokens saved | 1,925 |
| Wrong answers served | 1: "How do I **disable** 2FA?" got the **enable** answer (similarity 0.884) |

That last row is the main lesson: embeddings capture the topic well and miss negation,
so no similarity threshold can tell "enable" from "disable" apart.

## Setup

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com),
and a Redis 8+ database with Search enabled (the [Redis Cloud free tier](https://redis.io/try-free) works).

```bash
ollama pull llama3.2:3b          # answers questions
ollama pull nomic-embed-text     # turns text into 768-number vectors

cp .env.example .env             # then put your REDIS_URL in .env
uv sync
```

## Run

```bash
uv run streamlit run dashboard.py   # the lab: ask questions, run batches, watch the dashboard
uv run compare_models.py            # benchmark the installed Ollama models (cold vs warm)
```

The terminal running Streamlit prints every routing decision:

```
[HIT]  semantic_cache     145ms   249 tok  sim 0.848  'I forgot how to change my login password.'
[MISS] llm               7593ms   415 tok  sim 0.791  'How do I reset my router?'
```

## Things to try

- Run the sample batch twice: the second run is all hits.
- Drag the **similarity threshold** to 0.75: "reset my router" starts getting the password answer.
  Drag it to 0.90: real paraphrases start missing.
- Use the **similarity playground** to compare any two sentences.
- Set the TTL to 30 s and watch entries expire.

## Layout

```
cachelab/
  llm.py             Ollama chat calls with timing (latency, model load, tokens/sec)
  embeddings.py      text → unit vector, cosine similarity
  exact_cache.py     normalize → hash → Redis hash + TTL
  semantic_cache.py  Redis vector index (HNSW, cosine) + KNN lookup + threshold
  pipeline.py        the router: none | exact | semantic | hybrid
  redis_client.py    shared connection + cache:stats counters
  metrics.py         per-request log (SQLite) and the hit/miss math
dashboard.py         Streamlit UI
compare_models.py    model speed comparison
```

## Choosing a threshold

Thresholds depend on the embedding model, so measure before you pick one.
With `nomic-embed-text` and the `search_query: ` prefix:

```
0.43  What is Redis?  vs  What is PostgreSQL?            different
0.79  reset my password  vs  reset my router             different   ← close
0.84  reset my password  vs  forgot my login password    paraphrase  ← lowest real paraphrase
0.88  enable 2FA  vs  disable 2FA                        opposite!
0.97  What is Redis?  vs  Can you explain what Redis is? paraphrase
```

The default is **0.82**. A commonly quoted 0.65 would match "reset my router" here.

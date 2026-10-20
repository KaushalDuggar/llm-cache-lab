"""Per-request log + the numbers the dashboard shows (hit rate, miss rate, time saved...).

Stored in a local SQLite file so it works before Redis is in the picture.
Each row is one question asked, and where its answer came from.
"""

import sqlite3
import time
from pathlib import Path

import pandas as pd

DB_PATH = Path(__file__).resolve().parent.parent / "metrics.db"

# source = where the answer came from:
#   "llm"            -> cache MISS, the model generated it
#   "exact_cache"    -> cache HIT on identical text      (step 2)
#   "semantic_cache" -> cache HIT on similar meaning     (step 4)
SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    ts               REAL    NOT NULL,
    question         TEXT    NOT NULL,
    answer           TEXT    NOT NULL,
    model            TEXT    NOT NULL,
    cache_mode       TEXT    NOT NULL,
    source           TEXT    NOT NULL,
    cache_hit        INTEGER NOT NULL,
    latency_ms       REAL    NOT NULL,
    load_ms          REAL    NOT NULL DEFAULT 0,
    output_tokens    INTEGER NOT NULL DEFAULT 0,
    similarity       REAL,
    matched_question TEXT
)
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def record(
    *,
    question: str,
    answer: str,
    model: str,
    cache_mode: str,
    source: str,
    latency_ms: float,
    load_ms: float = 0.0,
    output_tokens: int = 0,
    similarity: float | None = None,
    matched_question: str | None = None,
) -> None:
    with _connect() as conn:
        conn.execute(
            """INSERT INTO requests (ts, question, answer, model, cache_mode, source, cache_hit,
                                     latency_ms, load_ms, output_tokens, similarity, matched_question)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (time.time(), question, answer, model, cache_mode, source, int(source != "llm"),
             latency_ms, load_ms, output_tokens, similarity, matched_question),
        )


def load() -> pd.DataFrame:
    with _connect() as conn:
        df = pd.read_sql_query("SELECT * FROM requests ORDER BY id", conn)
    df["ts"] = pd.to_datetime(df["ts"], unit="s")
    return df


def reset() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM requests")
        conn.execute("DELETE FROM sqlite_sequence WHERE name = 'requests'")


def summarize(df: pd.DataFrame) -> dict:
    total = len(df)
    hits = df[df["cache_hit"] == 1]
    misses = df[df["cache_hit"] == 0]

    avg_miss_ms = misses["latency_ms"].mean() if len(misses) else 0.0
    avg_hit_ms = hits["latency_ms"].mean() if len(hits) else 0.0

    return {
        "requests": total,
        "hits": len(hits),
        "misses": len(misses),
        "hit_rate": len(hits) / total if total else 0.0,
        "miss_rate": len(misses) / total if total else 0.0,
        "avg_hit_ms": avg_hit_ms,
        "avg_miss_ms": avg_miss_ms,
        # Every hit would otherwise have cost roughly one average miss.
        "time_saved_ms": max(len(hits) * avg_miss_ms - hits["latency_ms"].sum(), 0.0),
        # On a hit we log the token count of the reused answer = tokens the LLM didn't generate.
        "tokens_generated": int(misses["output_tokens"].sum()),
        "tokens_saved": int(hits["output_tokens"].sum()),
    }

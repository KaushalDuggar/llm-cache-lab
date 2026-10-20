"""Text -> vector (a list of 768 numbers) using a local Ollama embedding model."""

import time

import numpy as np
import ollama

from cachelab.llm import DEFAULT_EMBED_MODEL

DIM = 768  # nomic-embed-text outputs 768 numbers per text

# nomic-embed-text is trained with task prefixes. Using the same one on both sides
# spread paraphrases (0.84-0.97) further from look-alike questions (~0.79) than no prefix did.
PREFIX = "search_query: "


def embed(text: str, model: str = DEFAULT_EMBED_MODEL) -> tuple[np.ndarray, float]:
    """Returns (unit-length float32 vector, milliseconds taken)."""
    start = time.perf_counter()
    vec = np.array(ollama.embed(model=model, input=PREFIX + text).embeddings[0], dtype=np.float32)
    ms = (time.perf_counter() - start) * 1000
    return vec / np.linalg.norm(vec), ms


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """1.0 = same direction (same meaning), ~0 = unrelated."""
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

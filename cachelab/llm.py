"""Thin wrapper around Ollama so every LLM call comes back with timing info."""

import time
from dataclasses import dataclass

import ollama

DEFAULT_CHAT_MODEL = "llama3.2:3b"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


@dataclass
class LLMResult:
    text: str
    model: str
    latency_ms: float     # wall-clock time for the whole call, as the user feels it
    load_ms: float        # part of latency spent loading the model into memory (cold start)
    prompt_tokens: int
    output_tokens: int

    @property
    def tokens_per_sec(self) -> float:
        gen_ms = self.latency_ms - self.load_ms
        return self.output_tokens / (gen_ms / 1000) if gen_ms > 0 else 0.0


def chat(question: str, model: str = DEFAULT_CHAT_MODEL) -> LLMResult:
    start = time.perf_counter()
    resp = ollama.chat(model=model, messages=[{"role": "user", "content": question}])
    latency_ms = (time.perf_counter() - start) * 1000

    return LLMResult(
        text=resp.message.content,
        model=model,
        latency_ms=latency_ms,
        load_ms=(resp.load_duration or 0) / 1e6,  # Ollama reports nanoseconds
        prompt_tokens=resp.prompt_eval_count or 0,
        output_tokens=resp.eval_count or 0,
    )


def list_chat_models() -> list[str]:
    """Installed models that can chat (embedding-only models are filtered out)."""
    names = [m.model for m in ollama.list().models]
    return sorted(n for n in names if "embed" not in n and "minilm" not in n)

"""Run the same prompts through each installed model and compare speed.

    uv run compare_models.py
    uv run compare_models.py --models llama3.2:3b qwen2.5:7b --prompt "What is Redis?"

Not logged to the dashboard. This is just for picking a model.
"""

import argparse

from cachelab import llm

DEFAULT_PROMPTS = [
    "What is Redis? Answer in two sentences.",
    "How do I reset my password on a typical website?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", help="defaults to every installed chat model")
    parser.add_argument("--prompt", action="append", help="repeatable; defaults to two sample prompts")
    args = parser.parse_args()

    models = args.models or llm.list_chat_models()
    prompts = args.prompt or DEFAULT_PROMPTS
    if not models:
        raise SystemExit("No chat models installed. Try: ollama pull llama3.2:3b")

    print(f"{'model':<22} {'run':<5} {'total ms':>9} {'load ms':>8} {'tokens':>7} {'tok/s':>6}  answer")
    print("-" * 100)
    for model in models:
        for i, prompt in enumerate(prompts):
            r = llm.chat(prompt, model)
            # The first call per model includes loading weights into memory: that's the "cold start".
            label = "cold" if i == 0 else "warm"
            preview = r.text.replace("\n", " ")[:40]
            print(f"{model:<22} {label:<5} {r.latency_ms:>9.0f} {r.load_ms:>8.0f} "
                  f"{r.output_tokens:>7} {r.tokens_per_sec:>6.1f}  {preview}...")
        print()


if __name__ == "__main__":
    main()

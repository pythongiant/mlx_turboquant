"""Sample 50 ShareGPT first-turn prompts spread across length buckets.

Streams RyokoAI/ShareGPT52K, extracts each conversation's first human turn, and
collects a length-diverse set (measured with the Qwen3 tokenizer) so the
throughput benchmark exercises short, medium, long, and very-long prefills.
Writes benchmarks/sharegpt_50.json (bundled for reproducibility).
"""

import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

TOKENIZER = "mlx-community/Qwen3-1.7B-bf16"
OUT = Path(__file__).resolve().parent / "sharegpt_50.json"

# (name, min_tokens, max_tokens, count) — target counts per bucket, ~50 total.
BUCKETS = [
    ("short", 8, 64, 13),
    ("medium", 64, 256, 13),
    ("long", 256, 1024, 12),
    ("xlong", 1024, 3072, 12),
]


def first_human_turn(conv):
    for turn in conv:
        if turn.get("from") in ("human", "user"):
            v = (turn.get("value") or "").strip()
            if v:
                return v
    return None


def main():
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ds = load_dataset("RyokoAI/ShareGPT52K", split="train", streaming=True)

    want = {b[0]: b[3] for b in BUCKETS}
    got = {b[0]: [] for b in BUCKETS}
    seen = 0

    for ex in ds:
        seen += 1
        prompt = first_human_turn(ex.get("conversations") or [])
        if not prompt:
            continue
        n = len(tok.encode(prompt))
        for name, lo, hi, _ in BUCKETS:
            if lo <= n < hi and len(got[name]) < want[name]:
                got[name].append({"id": ex.get("id"), "n_tokens": n, "prompt": prompt})
                break
        if all(len(got[b[0]]) >= b[3] for b in BUCKETS):
            break
        if seen > 200_000:
            break

    examples = []
    for name, *_ in BUCKETS:
        examples.extend(got[name])
    examples.sort(key=lambda e: e["n_tokens"])

    OUT.write_text(json.dumps({"tokenizer": TOKENIZER, "examples": examples}, indent=1))
    lengths = [e["n_tokens"] for e in examples]
    print(f"wrote {len(examples)} examples to {OUT.name}  (scanned {seen} convos)")
    print(f"token lengths: min={min(lengths)} median={sorted(lengths)[len(lengths)//2]} "
          f"max={max(lengths)}")
    for name, lo, hi, _ in BUCKETS:
        print(f"  {name:6} [{lo:>4},{hi:>4}): {len(got[name])}")


if __name__ == "__main__":
    main()

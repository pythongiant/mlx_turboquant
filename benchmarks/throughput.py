"""Throughput & latency benchmark vs MLX LM over 50 varied-length ShareGPT prompts.

For each configuration we stream-generate a fixed number of tokens per prompt and
record, per prompt:
  * prefill throughput   (prompt tokens / sec)   -> also gives TTFT
  * decode  throughput   (generated tokens / sec) -> also gives TPOT
We report medians across the 50 prompts and speedup ratios vs the MLX LM
baselines (unquantized bf16 and MLX's own affine 4-bit).

Configurations (weights, KV cache):
  mlx-bf16            unquantized MLX LM                    (baseline)
  mlx-4bit            MLX LM affine 4-bit                   (MLX LM quantized baseline)
  turboquant-4bit     rotation + affine 4-bit               (this adapter, weights)
  turboquant-4bit-kv4 + rotated 4-bit TurboQuant KV cache   (this adapter, full)

Run:  python benchmarks/throughput.py --model mlx-community/Qwen3-1.7B-bf16
"""

import argparse
import gc
import json
import statistics as st
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

DATA = Path(__file__).resolve().parent / "sharegpt_50.json"
CONFIGS = [
    "mlx-bf16",
    "mlx-4bit",
    "turboquant-4bit",
    "turboquant-4bit-kv4",
    "turboquant-4bit-kv4-qjl",
]


def _model_gb(model):
    return sum(p.nbytes for _, p in tree_flatten(model.parameters())) / 1e9


def _free(model):
    del model
    gc.collect()
    try:
        mx.clear_cache()
    except Exception:
        pass


def build(config, model_id):
    from mlx_lm import load

    model, tok = load(model_id)  # always start from bf16
    cache_factory = lambda: None

    def is_linear(p, m):
        return isinstance(m, nn.Linear) and "lm_head" not in p

    if config == "mlx-bf16":
        pass
    elif config == "mlx-4bit":
        nn.quantize(model, group_size=64, bits=4, class_predicate=is_linear)
    elif config.startswith("turboquant-4bit"):
        from mlx_turboquant.quant_linear import turboquant_quantize

        turboquant_quantize(model, bits=4, group_size=64, mode="affine")
        if config.startswith("turboquant-4bit-kv4"):
            from mlx_turboquant.kv_cache import make_prompt_cache as mpc

            qjl = config.endswith("qjl")
            cache_factory = lambda: mpc(model, kv_bits=4, kv_group_size=64, qjl=qjl)
    else:
        raise ValueError(config)

    mx.eval(model.parameters())
    return model, tok, cache_factory


def measure(model, tok, cache_factory, prompt, max_tokens):
    from mlx_lm import stream_generate

    messages = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(messages, add_generation_prompt=True)
    last = None
    for resp in stream_generate(
        model, tok, text, max_tokens=max_tokens, prompt_cache=cache_factory()
    ):
        last = resp
    return last


def _kv_gb(model, cache_factory, prompt_tokens=2048):
    """Total KV-cache bytes after a fixed-length context (0 for fp16-KV configs)."""
    cache = cache_factory()
    if cache is None:
        return 0.0
    ids = mx.arange(prompt_tokens, dtype=mx.uint32)[None, :] % 1000
    model(ids, cache=cache)
    mx.eval([c.state for c in cache])
    return sum(c.nbytes for c in cache) / 1e9


def run_config(config, model_id, examples, max_tokens):
    model, tok, cache_factory = build(config, model_id)
    model_gb = _model_gb(model)
    kv_gb = _kv_gb(model, cache_factory)

    # Warmup (compiles kernels, warms caches) — not timed.
    measure(model, tok, cache_factory, "Hello, how are you?", 8)

    prefill_tps, decode_tps, ttft_ms, tpot_ms = [], [], [], []
    for ex in examples:
        r = measure(model, tok, cache_factory, ex["prompt"], max_tokens)
        if r is None or r.generation_tps <= 0 or r.prompt_tps <= 0:
            continue
        prefill_tps.append(r.prompt_tps)
        decode_tps.append(r.generation_tps)
        ttft_ms.append(1000.0 * r.prompt_tokens / r.prompt_tps)
        tpot_ms.append(1000.0 / r.generation_tps)

    _free(model)
    return {
        "config": config,
        "n": len(decode_tps),
        "prefill_tps": st.median(prefill_tps),
        "decode_tps": st.median(decode_tps),
        "ttft_ms": st.median(ttft_ms),
        "tpot_ms": st.median(tpot_ms),
        "model_gb": model_gb,
        "kv_gb_2k": kv_gb,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-1.7B-bf16")
    ap.add_argument("--max-tokens", type=int, default=64, dest="max_tokens")
    ap.add_argument("--configs", nargs="*", default=CONFIGS)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "throughput_results.json"))
    args = ap.parse_args()

    data = json.loads(DATA.read_text())
    examples = data["examples"]
    lengths = [e["n_tokens"] for e in examples]
    print(f"model={args.model}  prompts={len(examples)} "
          f"(tok len min={min(lengths)}/med={sorted(lengths)[len(lengths)//2]}/max={max(lengths)})  "
          f"decode={args.max_tokens} tokens\n")

    rows = []
    for cfg in args.configs:
        r = run_config(cfg, args.model, examples, args.max_tokens)
        rows.append(r)
        print(f"  {cfg:24} prefill={r['prefill_tps']:8.1f}  decode={r['decode_tps']:6.1f}  "
              f"TTFT={r['ttft_ms']:7.1f}ms  TPOT={r['tpot_ms']:5.1f}ms  "
              f"weights={r['model_gb']:.2f}GB  KV@2k={r['kv_gb_2k']:.3f}GB")

    base = {r["config"]: r for r in rows}
    Path(args.out).write_text(json.dumps(rows, indent=1))

    def ratio(cfg, ref, key, invert=False):
        if cfg not in base or ref not in base:
            return None
        a, b = base[cfg][key], base[ref][key]
        return (b / a) if invert else (a / b)

    def fmt(x):
        return f"{x:.2f}×" if x is not None else "  n/a"

    if "mlx-bf16" in base:
        print("\nSpeedup vs mlx-bf16 (×, >1 = faster):")
        for cfg in args.configs:
            if cfg == "mlx-bf16" or cfg not in base:
                continue
            print(f"  {cfg:24} decode {fmt(ratio(cfg,'mlx-bf16','decode_tps'))}  "
                  f"TPOT {fmt(ratio(cfg,'mlx-bf16','tpot_ms',invert=True))}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

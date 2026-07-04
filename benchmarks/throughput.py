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

DATA = Path(__file__).resolve().parent / "sharegpt_50.json"
CONFIGS = ["mlx-bf16", "mlx-4bit", "turboquant-4bit", "turboquant-4bit-kv4"]


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
        if config == "turboquant-4bit-kv4":
            from mlx_turboquant.kv_cache import make_prompt_cache as mpc

            cache_factory = lambda: mpc(model, kv_bits=4, kv_group_size=64)
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


def run_config(config, model_id, examples, max_tokens):
    model, tok, cache_factory = build(config, model_id)

    # Warmup (compiles kernels, warms caches) — not timed.
    measure(model, tok, cache_factory, "Hello, how are you?", 8)

    prefill_tps, decode_tps, ttft_ms, tpot_ms, peak = [], [], [], [], 0.0
    for ex in examples:
        r = measure(model, tok, cache_factory, ex["prompt"], max_tokens)
        if r is None or r.generation_tps <= 0 or r.prompt_tps <= 0:
            continue
        prefill_tps.append(r.prompt_tps)
        decode_tps.append(r.generation_tps)
        ttft_ms.append(1000.0 * r.prompt_tokens / r.prompt_tps)
        tpot_ms.append(1000.0 / r.generation_tps)
        peak = max(peak, getattr(r, "peak_memory", 0.0))

    _free(model)
    return {
        "config": config,
        "n": len(decode_tps),
        "prefill_tps": st.median(prefill_tps),
        "decode_tps": st.median(decode_tps),
        "ttft_ms": st.median(ttft_ms),
        "tpot_ms": st.median(tpot_ms),
        "peak_gb": peak,
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
        print(f"  {cfg:20} prefill={r['prefill_tps']:8.1f} tok/s  "
              f"decode={r['decode_tps']:6.1f} tok/s  TTFT={r['ttft_ms']:7.1f} ms  "
              f"TPOT={r['tpot_ms']:5.1f} ms  peak={r['peak_gb']:.2f} GB")

    base = {r["config"]: r for r in rows}
    Path(args.out).write_text(json.dumps(rows, indent=1))

    def ratio(cfg, ref, key, invert=False):
        if cfg not in base or ref not in base:
            return None
        a, b = base[cfg][key], base[ref][key]
        return (b / a) if invert else (a / b)

    print("\nSpeedup vs mlx-bf16 (×, >1 = faster):")
    for cfg in args.configs:
        if cfg == "mlx-bf16":
            continue
        print(f"  {cfg:20} decode {ratio(cfg,'mlx-bf16','decode_tps'):.2f}×  "
              f"prefill {ratio(cfg,'mlx-bf16','prefill_tps'):.2f}×  "
              f"TPOT {ratio(cfg,'mlx-bf16','tpot_ms',invert=True):.2f}×")
    print("\nSpeedup vs mlx-4bit (MLX LM's own quantization):")
    for cfg in ("turboquant-4bit", "turboquant-4bit-kv4"):
        if cfg in base and "mlx-4bit" in base:
            print(f"  {cfg:20} decode {ratio(cfg,'mlx-4bit','decode_tps'):.2f}×  "
                  f"prefill {ratio(cfg,'mlx-4bit','prefill_tps'):.2f}×")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()

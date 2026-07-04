"""KV-cache memory footprint vs context length: TurboQuant vs MLX affine.

The honest comparison is *iso-quality*: MLX's affine KV is unusable at 4-bit
(ppl ~31, see kv_quality.py), so to stay near fp16 quality affine needs **8-bit**
KV, whereas TurboQuant is near-neutral at **4-bit** and usable at **3-bit+QJL**.
So at matched quality TurboQuant's KV cache is ~2x smaller.

We measure the per-layer, per-token KV bytes of each cache directly (feed a dummy
step through it and read `nbytes`), then extrapolate linearly across context
length and layers. Emits kv_memory.json for the plotting script.
"""

import json
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_reduce
from transformers import AutoConfig

from mlx_lm.models.cache import KVCache, QuantizedKVCache

from mlx_turboquant.kv_cache import TurboQuantKVCache

MODEL = "mlx-community/Qwen3-1.7B-bf16"
OUT = Path(__file__).resolve().parent / "kv_memory.json"
PROBE_LEN = 4096  # divisible by the cache step (256) -> exact per-token bytes


def cache_bytes(cache):
    keys, values = cache.keys, cache.values
    total = tree_reduce(lambda a, x: a + x.nbytes, (keys, values), 0)
    if getattr(cache, "qjl", False) and getattr(cache, "sketch", None) is not None:
        total += cache.sketch.nbytes + cache.rnorm.nbytes
    return total


# (label, quality note, ppl, factory)
CONFIGS = [
    ("fp16 KV", "exact", 2.93, lambda: KVCache()),
    ("MLX affine 8-bit", "neutral", 2.91, lambda: QuantizedKVCache(group_size=64, bits=8)),
    ("MLX affine 4-bit", "broken", 31.4, lambda: QuantizedKVCache(group_size=64, bits=4)),
    ("TurboQuant 4-bit", "neutral", 3.16, lambda: TurboQuantKVCache(group_size=64, bits=4)),
    ("TurboQuant 3-bit + QJL", "usable", 4.82,
     lambda: TurboQuantKVCache(group_size=64, bits=3, qjl=True)),
]


def main():
    cfg = AutoConfig.from_pretrained(MODEL)
    tc = getattr(cfg, "text_config", cfg)
    n_layers = tc.num_hidden_layers
    n_kv = getattr(tc, "num_key_value_heads", tc.num_attention_heads)
    head_dim = getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads)
    print(f"{MODEL}: layers={n_layers} kv_heads={n_kv} head_dim={head_dim}\n")

    k = mx.zeros((1, n_kv, PROBE_LEN, head_dim), dtype=mx.float16)
    v = mx.zeros((1, n_kv, PROBE_LEN, head_dim), dtype=mx.float16)

    rows = []
    for label, note, ppl, factory in CONFIGS:
        c = factory()
        c.update_and_fetch(k, v)
        mx.eval(c.state)
        per_tok = cache_bytes(c) / PROBE_LEN * n_layers  # bytes / token (all layers)
        rows.append({"label": label, "note": note, "ppl": ppl,
                     "bytes_per_token": per_tok, "bits_per_channel": per_tok /
                     (2 * n_layers * n_kv * head_dim) * 8})  # K+V channels
        print(f"  {label:24} {note:8} ppl={ppl:6.2f}  {per_tok/1024:7.2f} KB/token")

    OUT.write_text(json.dumps({"model": MODEL, "n_layers": n_layers,
                               "n_kv_heads": n_kv, "head_dim": head_dim,
                               "configs": rows}, indent=1))
    print(f"\nwrote {OUT}")
    # Quick footprint table at a few context lengths.
    print("\nKV cache size (GB):")
    print(f"  {'context':>8}  " + "  ".join(f"{r['label'][:14]:>14}" for r in rows))
    for L in (4096, 16384, 32768, 131072):
        cells = "  ".join(f"{r['bytes_per_token']*L/1e9:>14.3f}" for r in rows)
        print(f"  {L:>8}  {cells}")


if __name__ == "__main__":
    main()

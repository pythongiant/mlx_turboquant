"""Render the KV-cache memory-vs-context chart from kv_memory.json.

Iso-quality framing: only configs that stay near fp16 quality are plotted as
solid lines (fp16, MLX affine 8-bit, TurboQuant 4-bit, TurboQuant 3-bit+QJL);
the point is that TurboQuant reaches affine-8-bit quality at ~half the memory.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

HERE = Path(__file__).resolve().parent
DATA = json.loads((HERE / "kv_memory.json").read_text())
OUT = HERE / "kv_memory.png"

# Surface + ink (dataviz light-mode tokens).
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#2b2b28", "#6b6b63", "#e6e5e0"
# Validated categorical hues: blue (hero), aqua, red (costly); gray = reference.
COL = {
    "fp16 KV": "#9a9a90",
    "MLX affine 8-bit": "#e34948",
    "TurboQuant 4-bit": "#2a78d6",
    "TurboQuant 3-bit + QJL": "#1baf7a",
}
# Only the quality-usable configs (affine 4-bit is broken -> table only).
SERIES = ["fp16 KV", "MLX affine 8-bit", "TurboQuant 4-bit", "TurboQuant 3-bit + QJL"]
XMAX = 131072
by = {c["label"]: c for c in DATA["configs"]}

fig, ax = plt.subplots(figsize=(9.4, 5.6), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)

# ShareGPT prompt-length band (the test set), near the origin.
ax.axvspan(13, 2631, color="#2a78d6", alpha=0.05, lw=0)
ax.text(2631, 0.2, "  ShareGPT prompts\n  (≤2.6k tok)", fontsize=8, color=MUTED, va="bottom")

label_pts = []
for name in SERIES:
    bpt = by[name]["bytes_per_token"]
    gb = bpt * XMAX / 1e9
    lw = 2.8 if name.startswith("TurboQuant") else 2.2
    ls = "--" if name == "fp16 KV" else "-"
    ax.plot([0, XMAX], [0, gb], color=COL[name], lw=lw, ls=ls,
            solid_capstyle="round", zorder=3)
    label_pts.append((gb, name, by[name]["ppl"]))

# Direct labels at the right end, de-collided (2-line labels need ~1 GB gap).
label_pts.sort()
prev = -1e9
for gb, name, ppl in label_pts:
    y = gb if gb - prev > 1.05 else prev + 1.05
    prev = y
    ax.text(XMAX * 1.015, y, f"{name}\n{gb:.1f} GB · ppl {ppl:.1f}",
            color=COL[name], fontsize=8.5, va="center", fontweight="bold")

# Iso-quality callout: affine-8bit vs TurboQuant-4bit at 128k.
a8 = by["MLX affine 8-bit"]["bytes_per_token"] * XMAX / 1e9
t4 = by["TurboQuant 4-bit"]["bytes_per_token"] * XMAX / 1e9
ax.annotate("", xy=(XMAX * 0.86, t4), xytext=(XMAX * 0.86, a8),
            arrowprops=dict(arrowstyle="<->", color=INK, lw=1.3))
ax.text(XMAX * 0.845, (a8 + t4) / 2,
        f"≈{a8 / t4:.1f}× less KV\nat matched quality", ha="right", va="center",
        fontsize=9.5, color=INK, fontweight="bold")

ax.set_xlim(0, XMAX)
ax.set_ylim(0, by["fp16 KV"]["bytes_per_token"] * XMAX / 1e9 * 1.08)
ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{int(x/1024)}k" if x else "0"))
ax.set_xticks([0, 32768, 65536, 98304, 131072])
ax.set_xlabel("context length (tokens)", fontsize=10.5, color=INK)
ax.set_ylabel("KV cache memory (GB)", fontsize=10.5, color=INK)
ax.set_title("KV cache memory vs context length — Qwen3-1.7B",
             fontsize=13.5, color=INK, fontweight="bold", pad=16, loc="left")
ax.text(0, 1.02, "At matched (near-fp16) quality, TurboQuant's rotated KV needs "
        "~half the memory of MLX affine.", transform=ax.transAxes,
        fontsize=9.5, color=MUTED)

for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.margins(x=0)
plt.subplots_adjust(right=0.74, left=0.09, top=0.84, bottom=0.12)
fig.savefig(OUT, facecolor=SURFACE, bbox_inches="tight")
print(f"wrote {OUT}")

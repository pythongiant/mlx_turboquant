"""``turboquant`` command-line interface: convert / generate / eval."""

from __future__ import annotations

import argparse
import sys


def _cmd_convert(args):
    from .convert import convert

    convert(
        args.model,
        args.out,
        bits=args.bits,
        group_size=args.group_size,
        rot_base_seed=args.seed,
        dtype=args.dtype,
        skip=tuple(args.skip),
        mode=args.mode,
        upload_repo=args.upload_repo,
    )


def _cmd_generate(args):
    from .patch import register

    register()
    from mlx_lm import generate, load

    model, tokenizer = load(args.model)
    messages = [{"role": "user", "content": args.prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    ) if getattr(tokenizer, "chat_template", None) else args.prompt

    prompt_cache = None
    if args.kv_bits is not None:
        # Use the TurboQuant (rotated) KV cache, not mlx-lm's plain affine one.
        from .kv_cache import make_prompt_cache

        prompt_cache = make_prompt_cache(
            model, kv_bits=args.kv_bits, kv_group_size=args.kv_group_size
        )

    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        verbose=True,
        prompt_cache=prompt_cache,
    )


def _cmd_eval(args):
    import runpy
    import sys

    # Delegate to the KV-quality benchmark (teacher-forced perplexity).
    from pathlib import Path

    bench = Path(__file__).resolve().parent.parent / "benchmarks" / "kv_quality.py"
    if not bench.exists():
        print("Benchmark script not found; install from source to run eval.")
        return
    sys.argv = ["kv_quality.py", "--model", args.model]
    runpy.run_path(str(bench), run_name="__main__")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="turboquant", description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("convert", help="Quantize a model with TurboQuant.")
    c.add_argument("--model", required=True, help="HF repo or local fp model path.")
    c.add_argument("--out", required=True, help="Output directory.")
    c.add_argument("--bits", type=int, default=4)
    c.add_argument("--group-size", type=int, default=64, dest="group_size")
    c.add_argument("--seed", type=int, default=0, help="Rotation base seed.")
    c.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "lut", "affine"],
        help="Scalar quantizer: 'lut' = non-uniform Lloyd-Max Metal kernel, "
        "'affine' = MLX built-in, 'auto' picks LUT for <=4-bit.",
    )
    c.add_argument("--dtype", default=None, choices=[None, "float16", "bfloat16", "float32"])
    c.add_argument("--skip", nargs="*", default=["lm_head"], help="Substrings of layers to leave unquantized.")
    c.add_argument("--upload-repo", default=None, dest="upload_repo")
    c.set_defaults(func=_cmd_convert)

    g = sub.add_parser("generate", help="Generate text with a TurboQuant model.")
    g.add_argument("--model", required=True)
    g.add_argument("--prompt", required=True)
    g.add_argument("--max-tokens", type=int, default=256, dest="max_tokens")
    g.add_argument("--kv-bits", type=int, default=None, dest="kv_bits")
    g.add_argument("--kv-group-size", type=int, default=64, dest="kv_group_size")
    g.set_defaults(func=_cmd_generate)

    e = sub.add_parser("eval", help="(stub) pointer to perplexity evaluation.")
    e.add_argument("--model", required=True)
    e.set_defaults(func=_cmd_eval)
    return p


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

"""Converts a k-diffusion training checkpoint to a slim inference checkpoint."""

import argparse
import json
from pathlib import Path
import sys

import torch
import safetensors.torch as safetorch


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", type=Path,
                   help="the training checkpoint to convert")
    p.add_argument("--config", type=Path,
                   help="override the checkpoint's configuration")
    p.add_argument("--output", "-o", type=Path,
                   help="the output slim checkpoint")
    p.add_argument("--dtype", type=str, choices=["fp32", "fp16", "bf16"], default="fp16",
                   help="the output dtype")
    args = p.parse_args()

    print(f"Loading training checkpoint {args.checkpoint}...", file=sys.stderr)
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    config = ckpt.get("config", None)
    model_ema = ckpt["model_ema"]
    del ckpt

    if args.config:
        config = json.loads(args.config.read_text())

    if config is None:
        raise ValueError("No configuration found in checkpoint and no override provided")

    dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    model_ema = {k: v.to(dtype) for k, v in model_ema.items()}

    output_path = args.output or args.checkpoint.with_suffix(".safetensors")
    metadata = {"config": json.dumps(config, indent=4)}
    print(f"Saving inference checkpoint to {output_path}...", file=sys.stderr)
    safetorch.save_file(model_ema, output_path, metadata=metadata)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Extracts the configuration file from a slim inference checkpoint."""

import argparse
from pathlib import Path
import sys

import k_diffusion as K


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("checkpoint", type=Path,
                   help="the inference checkpoint to extract the configuration from")
    p.add_argument("--output", "-o", type=Path,
                   help="the output configuration file")
    args = p.parse_args()

    print(f"Loading inference checkpoint {args.checkpoint}...", file=sys.stderr)
    metadata = K.utils.get_safetensors_metadata(args.checkpoint)
    if "config" not in metadata:
        raise ValueError("No configuration found in checkpoint")

    output_path = args.output or args.checkpoint.with_suffix(".json")

    print(f"Saving configuration to {output_path}...", file=sys.stderr)
    output_path.write_text(metadata["config"])


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Evaluate one explicitly chosen checkpoint and split; never run from training."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from qtra.data.loading import cdvqa_dataset, qag_dataset
from qtra.evaluation import evaluate
from qtra.factory import build_model, load_weights, read_config
from qtra.training import set_seed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--clip-pretrained", required=True)
    parser.add_argument("--split", choices=["Val", "Test", "Test2"], required=True)
    parser.add_argument("--data-root", help="QAG directory containing json/, image/ and mask/")
    parser.add_argument("--cdvqa-root")
    parser.add_argument("--second-root")
    parser.add_argument("--output", required=True, help="New JSON file, separate from training logs")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--limit", type=int, help="Debug only: first N samples, not a benchmark result")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an evaluation result: {output}")
    config = read_config(args.config)
    set_seed(42)
    device = torch.device(args.device)
    precision = args.precision or config["training"]["precision"]
    model = build_model(config["model"], args.clip_pretrained)
    load_weights(model, args.checkpoint)
    model.to(device)
    if config["task"] == "qag":
        if not args.data_root or args.split == "Test2":
            parser.error("QAG requires --data-root and split Val or Test")
        dataset = qag_dataset(args.data_root, args.split)
    else:
        if not args.cdvqa_root or not args.second_root:
            parser.error("CDVQA requires --cdvqa-root and --second-root")
        dataset = cdvqa_dataset(args.cdvqa_root, args.second_root, args.split)
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    metrics = evaluate(model, loader, device, config["task"], config.get("mask_threshold", 0.35), precision, args.split)
    with open(args.checkpoint, "rb") as handle:
        checksum = hashlib.file_digest(handle, "sha256").hexdigest()
    result = {
        "split": args.split, "checkpoint_sha256": checksum,
        "precision": precision, "debug_subset": args.limit is not None,
        "config": config, "metrics": metrics,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

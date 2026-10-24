#!/usr/bin/env python3
"""Assign precomputed embedding rows to stratified train/val/test splits.

Example:
    python embedding_ml/split.py --input data/embedding_csv/embeddings.csv \
        --ratios 0.7 0.15 0.15 --seed 42

Writes embeddings_split.csv and train.csv, val.csv, test.csv under --output-dir.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


SPLITS = ("train", "val", "test")


def split_counts(count: int, ratios: tuple[float, float, float]) -> list[int]:
    """Largest-remainder allocation, preserving the exact class total."""
    raw = [count * ratio for ratio in ratios]
    allocated = [int(value) for value in raw]
    remainder = count - sum(allocated)
    order = sorted(range(3), key=lambda i: (raw[i] - allocated[i], -i), reverse=True)
    for index in order[:remainder]:
        allocated[index] += 1
    return allocated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/embedding_csv/embeddings.csv"),
                        help="CSV embeddings đã tạo bởi extract_embedding.py")
    parser.add_argument("--output-dir", type=Path, default=Path("data/embedding_csv/splits"))
    parser.add_argument("--ratios", nargs=3, type=float, default=(0.7, 0.15, 0.15),
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = tuple(args.ratios)
    if any(ratio < 0 for ratio in ratios) or sum(ratios) <= 0:
        raise ValueError("Tỉ lệ phải không âm và tổng lớn hơn 0.")
    total = sum(ratios)
    ratios = tuple(ratio / total for ratio in ratios)

    with args.input.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if not reader.fieldnames or not {"label", "label_id"}.issubset(reader.fieldnames):
            raise ValueError("CSV cần có cột label và label_id.")
        fields = list(reader.fieldnames)
        rows = list(reader)

    if not rows:
        raise ValueError(f"CSV không có dữ liệu: {args.input}")
    if "dataset" not in fields:
        fields.append("dataset")

    by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)

    rng = random.Random(args.seed)
    for label in sorted(by_label):
        label_rows = by_label[label]
        rng.shuffle(label_rows)
        counts = split_counts(len(label_rows), ratios)
        start = 0
        for split, count in zip(SPLITS, counts):
            for row in label_rows[start:start + count]:
                row["dataset"] = split
            start += count

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {split: args.output_dir / f"{split}.csv" for split in SPLITS}
    outputs["all"] = args.output_dir / "embeddings_split.csv"
    handles = {name: path.open("w", encoding="utf-8-sig", newline="")
               for name, path in outputs.items()}
    try:
        writers = {name: csv.DictWriter(handle, fieldnames=fields) for name, handle in handles.items()}
        for writer in writers.values():
            writer.writeheader()
        for row in rows:
            writers["all"].writerow(row)
            writers[row["dataset"]].writerow(row)
    finally:
        for handle in handles.values():
            handle.close()

    print(f"Seed: {args.seed}; ratios (train/val/test): {ratios}")
    for split in SPLITS:
        print(f"{split}: {sum(row['dataset'] == split for row in rows):,} → {outputs[split]}")
    print(f"All assignments: {outputs['all']}")


if __name__ == "__main__":
    main()

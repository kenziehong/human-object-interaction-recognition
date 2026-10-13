#!/usr/bin/env python3
"""Extract frozen ImageNet features from ResNet50, EfficientNet-B0 and ViT-B/16.

Expected dataset structure:
    data/vcoco_6class_crops/
        train/<label>/*.{jpg,png,...}
        val/<label>/*.{jpg,png,...}
        test/<label>/*.{jpg,png,...}

Example:
    python extract_embedding.py \
        --data-dir data/vcoco_6class_crops \
        --output-dir data/embedding_csv \
        --batch-size 32

The script writes one CSV containing one row per image, with a separate JSON
array column for each backbone's feature vector.
"""

from __future__ import annotations

import argparse
import csv
from contextlib import ExitStack
import gc
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


SPLITS: Tuple[str, ...] = ("train", "val", "test")
IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

MODEL_CONFIGS = {
    "resnet50": {
        "weights_name": "ResNet50_Weights.DEFAULT",
        "feature_layer": "avgpool_before_fc",
        "embedding_dim": 2048,
    },
    "efficientnet_b0": {
        "weights_name": "EfficientNet_B0_Weights.DEFAULT",
        "feature_layer": "avgpool_before_classifier",
        "embedding_dim": 1280,
    },
    "vit_b_16": {
        "weights_name": "ViT_B_16_Weights.DEFAULT",
        "feature_layer": "cls_token_before_head",
        "embedding_dim": 768,
    },
}

CSV_FIELDS = [
    "stt",
    "image_path",
    "relative_path",
    "filename",
    "dataset",
    "label",
    "label_id",
    "model",
    "weights",
    "resnet50_embedding",
    "efficientnet_b0_embedding",
    "vit_b_16_embedding",
]


def choose_device(requested: str = "auto") -> torch.device:
    """Choose an available inference device (MPS first for Apple Silicon)."""
    if requested == "auto":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Using device: {device}")
        return device

    device = torch.device(requested)
    if device.type == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("Bạn chọn device=mps nhưng MPS không khả dụng trong môi trường hiện tại.")
    elif device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Bạn chọn device=cuda nhưng CUDA không khả dụng trong môi trường hiện tại.")

    print(f"Using device: {device}")
    return device


def collect_image_records(data_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Scan train/val/test label directories and assign stable IDs to labels."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy thư mục dataset: {data_dir}")

    records: List[Dict[str, Any]] = []
    all_labels = set()
    stt = 1

    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            raise FileNotFoundError(
                f"Thiếu thư mục '{split}'. Dataset cần có đủ train/, val/, test/: {split_dir}"
            )

        label_dirs = sorted(p for p in split_dir.iterdir() if p.is_dir())
        if not label_dirs:
            warnings.warn(f"Không tìm thấy thư mục nhãn nào trong {split_dir}")

        for label_dir in label_dirs:
            label = label_dir.name
            all_labels.add(label)
            image_paths = sorted(
                p for p in label_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
            )

            for image_path in image_paths:
                records.append(
                    {
                        "stt": stt,
                        "image_path": str(image_path.resolve()),
                        "relative_path": image_path.relative_to(data_dir).as_posix(),
                        "filename": image_path.name,
                        "dataset": split,
                        "label": label,
                    }
                )
                stt += 1

    label_to_id = {label: idx for idx, label in enumerate(sorted(all_labels))}
    for record in records:
        record["label_id"] = label_to_id[record["label"]]

    if not records:
        raise RuntimeError(
            f"Không tìm thấy ảnh hợp lệ trong {data_dir}. Hãy kiểm tra cấu trúc và phần mở rộng ảnh."
        )

    print(f"Found {len(records):,} images.")
    counts = Counter(record["dataset"] for record in records)
    for split in SPLITS:
        print(f"  {split}: {counts.get(split, 0):,} images")
    print(f"Labels ({len(label_to_id)}): {label_to_id}")
    return records, label_to_id


def build_model_and_transform(
    model_name: str,
    preprocessing: str = "weights",
) -> Tuple[nn.Module, Any, int]:
    """Create a pretrained feature extractor and its preprocessing transform.

    preprocessing='weights' uses the canonical transform associated with each
    torchvision weight set. preprocessing='common' uses Resize(256),
    CenterCrop(224), and ImageNet mean/std for all three models.
    """
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()
    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)
        model.classifier = nn.Identity()
    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        model.heads = nn.Identity()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if preprocessing == "weights":
        image_transform = weights.transforms()
    elif preprocessing == "common":
        image_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        raise ValueError(f"Unsupported preprocessing: {preprocessing}")

    model.eval()
    model.requires_grad_(False)
    expected_dim = MODEL_CONFIGS[model_name]["embedding_dim"]
    return model, image_transform, expected_dim


class ImageEmbeddingDataset(Dataset):
    """Image dataset that returns a transformed image and its metadata."""

    def __init__(self, records: Sequence[Dict[str, Any]], image_transform: Any):
        self.records = records
        self.image_transform = image_transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        try:
            with Image.open(record["image_path"]) as image:
                image = image.convert("RGB")
                image_tensor = self.image_transform(image)
            return image_tensor, record
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            warnings.warn(f"Bỏ qua ảnh lỗi '{record['image_path']}': {exc}")
            return None


def collate_valid_images(batch):
    """Batch images while dropping corrupted files returned as None."""
    valid_samples = [sample for sample in batch if sample is not None]
    if not valid_samples:
        return None
    images, records = zip(*valid_samples)
    return torch.stack(images, dim=0), list(records)


def extract_batch_embeddings(
    model: nn.Module,
    images: torch.Tensor,
    model_name: str,
    device: torch.device,
) -> torch.Tensor:
    """Run inference and return one vector per image."""
    images = images.to(device)
    with torch.inference_mode():
        embeddings = model(images)

    if embeddings.ndim != 2:
        raise RuntimeError(
            f"Mong đợi embedding dạng [batch, dim], nhận được shape={tuple(embeddings.shape)}"
        )

    expected_dim = MODEL_CONFIGS[model_name]["embedding_dim"]
    if embeddings.shape[1] != expected_dim:
        raise RuntimeError(
            f"Kích thước embedding {model_name} không đúng: "
            f"mong đợi {expected_dim}, nhận được {embeddings.shape[1]}"
        )
    return embeddings.float().cpu()


def save_model_embeddings_csv(
    model_name: str,
    records: Sequence[Dict[str, Any]],
    output_dir: Path,
    device: torch.device,
    batch_size: int = 32,
    num_workers: int = 0,
    preprocessing: str = "weights",
) -> Path:
    """Extract features for one backbone and save a temporary CSV."""
    print(f"\n=== Extracting: {model_name} ===")
    model, image_transform, embedding_dim = build_model_and_transform(
        model_name=model_name,
        preprocessing=preprocessing,
    )
    model = model.to(device)

    dataset = ImageEmbeddingDataset(records, image_transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_valid_images,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f".{model_name}_embeddings.tmp.csv"
    config = MODEL_CONFIGS[model_name]

    processed = 0
    with output_path.open("w", newline="", encoding="utf-8-sig") as csv_file:
        embedding_field = f"{model_name}_embedding"
        fields = [
            "stt", "image_path", "relative_path", "filename", "dataset",
            "label", "label_id", "model", "weights", embedding_field,
        ]
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()

        for batch in loader:
            if batch is None:
                continue
            images, batch_records = batch
            batch_embeddings = extract_batch_embeddings(
                model=model,
                images=images,
                model_name=model_name,
                device=device,
            ).numpy()

            for record, embedding in zip(batch_records, batch_embeddings):
                # JSON arrays can be read later with json.loads(row[field]).
                embedding_json = json.dumps(
                    np.round(embedding.astype(np.float32), 8).tolist(),
                    separators=(",", ":"),
                    ensure_ascii=False,
                )
                row = {
                    **record,
                    "model": model_name,
                    "weights": config["weights_name"],
                    embedding_field: embedding_json,
                }
                writer.writerow(row)
                processed += 1

            if processed and (processed % (batch_size * 10) == 0):
                print(f"  Processed {processed:,}/{len(records):,} images...")

    print(f"Saved: {output_path}")
    print(f"Rows written: {processed:,}")

    # Free one backbone before loading the next (useful on Apple Silicon laptops).
    del model, dataset, loader, image_transform
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()

    return output_path


def merge_embedding_csvs(paths: Dict[str, Path], output_path: Path) -> None:
    """Merge per-backbone temporary CSVs into one complete row per image."""
    base_fields = CSV_FIELDS[:7]
    with ExitStack() as stack:
        readers = {
            name: csv.DictReader(stack.enter_context(path.open(encoding="utf-8-sig", newline="")))
            for name, path in paths.items()
        }
        with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=CSV_FIELDS)
            writer.writeheader()
            iterators = {name: iter(reader) for name, reader in readers.items()}
            while True:
                rows = {name: next(rows, None) for name, rows in iterators.items()}
                if all(row is None for row in rows.values()):
                    break
                if any(row is None for row in rows.values()):
                    raise RuntimeError("Các file embedding không có cùng số dòng.")
                anchor = next(iter(rows.values()))
                if any(any(row[field] != anchor[field] for field in base_fields) for row in rows.values()):
                    raise RuntimeError("Thứ tự ảnh giữa các file embedding không khớp.")
                combined = {field: anchor[field] for field in base_fields}
                combined["model"] = ",".join(rows)
                combined["weights"] = json.dumps(
                    {name: row["weights"] for name, row in rows.items()},
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
                for name, row in rows.items():
                    combined[f"{name}_embedding"] = row[f"{name}_embedding"]
                writer.writerow(combined)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract pretrained image embeddings and save them to CSV."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/vcoco_6class_crops"),
        help="Dataset root containing train/, val/, and test/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/embedding_csv"),
        help="Directory where the model CSV files will be written.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_CONFIGS.keys()),
        default=["resnet50", "efficientnet_b0", "vit_b_16"],
        help="One or more backbones to extract.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size. Reduce it if memory is insufficient.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers; 0 is a safe default on macOS.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "mps", "cuda", "cpu"),
        default="auto",
        help="Inference device. auto prefers MPS, then CUDA, then CPU.",
    )
    parser.add_argument(
        "--preprocessing",
        choices=("weights", "common"),
        default="weights",
        help=(
            "weights: recommended preprocessing for each pretrained weight set; "
            "common: use Resize(256)+CenterCrop(224)+ImageNet normalization for all."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size phải lớn hơn 0.")
    if args.num_workers < 0:
        raise ValueError("--num-workers không được âm.")

    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    device = choose_device(args.device)
    records, _label_to_id = collect_image_records(data_dir)

    temporary_paths = {}
    for model_name in args.models:
        temporary_paths[model_name] = save_model_embeddings_csv(
            model_name=model_name,
            records=records,
            output_dir=output_dir,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            preprocessing=args.preprocessing,
        )

    output_path = output_dir / "embeddings.csv"
    merge_embedding_csvs(temporary_paths, output_path)
    for path in temporary_paths.values():
        path.unlink(missing_ok=True)
    print(f"Saved complete image rows: {output_path}")

    print("\nAll requested embeddings have been exported.")


if __name__ == "__main__":
    main()

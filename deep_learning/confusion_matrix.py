#!/usr/bin/env python3
"""Evaluate a deep-learning checkpoint on the CSV test split and save a confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from config import CLASS_NAMES, ROOT, select_device
from data import InteractionImageDataset, load_manifest, make_transform
from model import build_model


DEFAULT_CHECKPOINT = (
    ROOT / "deep_learning/outputs/resnet50_th2/resnet50_th2_seed42_03/best.pt"
)
DEFAULT_CSV = ROOT / "data/train_test_split/labels_split_unk3000.csv"
DEFAULT_IMAGE_ROOT = ROOT / "data/dataset_6class_crops_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT,
                        help="Checkpoint best.pt cần đánh giá.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV,
                        help="CSV manifest có cột split.")
    parser.add_argument("--image-root", type=Path, default=DEFAULT_IMAGE_ROOT,
                        help="Thư mục gốc chứa các crop trong crop_path.")
    parser.add_argument("--output", type=Path, default=None,
                        help="File ảnh đầu ra (mặc định: confusion_matrix_test.png cạnh checkpoint).")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size suy luận.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--normalize", choices=("true", "pred", "all"),
                        help="Chuẩn hóa theo nhãn thật, nhãn dự đoán hoặc toàn bộ mẫu.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {args.model}")
    if args.num_workers < 0:
        raise ValueError("num-workers không được âm.")

    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError("Checkpoint thiếu state_dict.")
    backbone = checkpoint.get("backbone")
    if backbone not in {"resnet50", "efficientnet_b0"}:
        raise ValueError(f"Backbone không được hỗ trợ trong checkpoint: {backbone}")
    class_names = tuple(checkpoint.get("class_names", CLASS_NAMES))
    if class_names != CLASS_NAMES:
        raise ValueError("Thứ tự class trong checkpoint không khớp với CSV/pipeline.")

    if args.batch_size < 1:
        raise ValueError("batch-size phải lớn hơn 0.")

    manifest = load_manifest(args.csv, args.image_root, check_images=False)
    dataset = InteractionImageDataset(manifest, "test", make_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=False)

    device = select_device(args.device)
    model = build_model(backbone, pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()

    y_true_batches = []
    y_pred_batches = []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            y_true_batches.append(labels.numpy())
            y_pred_batches.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(y_true_batches)
    y_pred = np.concatenate(y_pred_batches)

    labels = list(range(len(class_names)))
    matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize=args.normalize)
    output_path = args.output or args.model.parent / "confusion_matrix_test.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(ax=ax, cmap="Blues", values_format=".2f" if args.normalize else "d",
                 colorbar=True, xticks_rotation=35)
    model_name = checkpoint.get("model_name", args.model.parent.name)
    ax.set_title(f"Confusion matrix: {model_name} (test, n={len(y_true):,})")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Model: {args.model}")
    print(f"Test samples: {len(y_true):,}; device: {device}")
    print(f"Confusion matrix image: {output_path}")
    print("\nClassification report (test):")
    print(classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                digits=4, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(matrix)


if __name__ == "__main__":
    main()

"""Shared configuration and command-line helpers for image experiments."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
CLASS_NAMES = ("hold", "ride", "eat/drink", "talk_on_phone", "work_on_computer", "unknown")
LABEL_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class ExperimentConfig:
    backbone: str = "resnet50"
    setup: str = "th1"
    seed: int = 42
    batch_size: int = 16
    max_epochs: int = 30
    patience: int = 6
    min_delta: float = 0.0
    classifier_lr: float = 1e-3
    backbone_lr: float = 1e-5
    weight_decay: float = 1e-4
    unfreeze_epoch: int = 3
    num_workers: int = 0
    device: str = "auto"
    pretrained: bool = True

    def to_dict(self) -> dict:
        values = asdict(self)
        values.update({"class_names": list(CLASS_NAMES), "image_size": IMAGE_SIZE,
                       "mean": IMAGENET_MEAN, "std": IMAGENET_STD})
        return values


def add_experiment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backbone", choices=("resnet50", "efficientnet_b0"), default="resnet50")
    parser.add_argument("--setup", choices=("th1", "th2", "th3"), default="th1")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--classifier-lr", type=float)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--unfreeze-epoch", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    parser.add_argument("--no-pretrained", action="store_true")


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    default_lr = 1e-4 if args.setup == "th3" else 1e-3
    return ExperimentConfig(
        backbone=args.backbone, setup=args.setup, seed=args.seed, batch_size=args.batch_size,
        max_epochs=args.epochs, patience=args.patience, min_delta=args.min_delta,
        classifier_lr=args.classifier_lr or default_lr, backbone_lr=args.backbone_lr,
        weight_decay=args.weight_decay, unfreeze_epoch=args.unfreeze_epoch,
        num_workers=args.num_workers, device=args.device, pretrained=not args.no_pretrained,
    )


def select_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    mps_available = torch.backends.mps.is_built() and torch.backends.mps.is_available()
    if requested == "mps":
        if not mps_available:
            raise RuntimeError("MPS không khả dụng. Kiểm tra Python arm64 và bản PyTorch macOS.")
        return torch.device("mps")
    return torch.device("mps" if mps_available else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def update_results_summary(outputs_dir: Path | None = None) -> Path:
    """Rebuild a readable summary of running and completed runs."""
    outputs_dir = outputs_dir or ROOT / "deep_learning/outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    runs = []
    for config_path in sorted(outputs_dir.glob("*/config.json")):
        run_dir = config_path.parent
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            metrics_path = run_dir / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
        except (OSError, json.JSONDecodeError):
            continue
        runs.append((run_dir, config, metrics))

    best_by_setup = {}
    for run_dir, config, metrics in runs:
        val = (config.get("latest_metrics", {}).get("val", {})
               if not config.get("training_complete") else metrics.get("val", {}))
        key = (config.get("backbone"), config.get("setup"))
        score = val.get("f1_macro", -1)
        loss = val.get("loss", float("inf"))
        current = best_by_setup.get(key)
        if current is None or (score, -loss) > current[0]:
            best_by_setup[key] = ((score, -loss), run_dir)

    lines = ["TỔNG HỢP KẾT QUẢ DEEP LEARNING",
             "Run tốt nhất trong mỗi backbone/TH được chọn theo val macro-F1; nếu hòa, ưu tiên val loss thấp hơn.",
             "Các metric được cập nhật trong lúc train; test chỉ xuất hiện sau khi đánh giá run thắng.", ""]
    if not runs:
        lines.append("Chưa có run.")
    for run_dir, config, metrics in runs:
        in_progress = not config.get("training_complete")
        latest = config.get("latest_metrics", {})
        val = latest.get("val", {}) if in_progress else metrics.get("val", {})
        train = latest.get("train", {}) if in_progress else metrics.get("train", {})
        test = metrics.get("test", {})
        key = (config.get("backbone"), config.get("setup"))
        is_best = best_by_setup[key][1] == run_dir
        lines.extend([
            "=" * 78,
            f"Cấu hình: {config.get('backbone')} | {str(config.get('setup', '')).upper()} | seed={config.get('seed')}",
            f"Run: {run_dir.name}" + ("  [TỐT NHẤT THEO VAL CHO CẤU HÌNH NÀY]" if is_best else ""),
            f"Trạng thái: {'đã đánh giá test' if test else ('đang huấn luyện' if in_progress else ('đã khóa, chưa đánh giá test' if config.get('locked') else 'chưa khóa'))}",
            ("Tham số: " + ", ".join(f"{name}={config.get(name)}" for name in (
                "pretrained", "batch_size", "max_epochs", "patience", "min_delta",
                "classifier_lr", "backbone_lr", "weight_decay", "unfreeze_epoch", "num_workers", "device_actual"))),
            f"Epoch tốt nhất: {config.get('best_epoch')}",
            (f"Tiến độ: epoch {config.get('last_epoch')}/{config.get('max_epochs')}"
             if in_progress and config.get("last_epoch") else ""),
            ("Train: " + _format_summary_metrics(train)),
            ("Val:   " + _format_summary_metrics(val)),
            ("Test:  " + (_format_summary_metrics(test) if test else "chưa có")),
            f"Checkpoint: {run_dir / 'best.pt'}",
            f"Chi tiết: {run_dir / 'metrics.json'} | {run_dir / 'config.json'} | {run_dir / 'report.txt'}",
            "",
        ])
    summary_path = outputs_dir / "results_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def _format_summary_metrics(metrics: dict) -> str:
    if not metrics:
        return "chưa có"
    fields = ("loss", "accuracy", "f1_macro", "f1_weighted", "balanced_accuracy")
    return ", ".join(f"{name}={metrics[name]:.4f}" for name in fields if name in metrics)

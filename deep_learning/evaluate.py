"""Lock a selected run or evaluate its checkpoint on test once."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import CLASS_NAMES, select_device, update_results_summary, write_json
from data import InteractionImageDataset, load_manifest, make_transform
from metrics import format_report
from model import build_model
from train import log_run, run_epoch


def evaluate_run(run_dir: Path, device_name: str = "auto", manifest=None) -> dict:
    """Evaluate a completed, locked run on test once and persist its metrics/report."""
    config_path = run_dir / "config.json"
    if not config_path.is_file() or not (run_dir / "best.pt").is_file():
        raise FileNotFoundError("Run cần có config.json và best.pt.")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not config.get("training_complete") or not config.get("locked"):
        raise ValueError("Chỉ đánh giá test sau khi train hoàn tất và run được khóa bằng --lock-run.")
    metrics_path = run_dir / "metrics.json"
    if metrics_path.is_file() and "test" in json.loads(metrics_path.read_text(encoding="utf-8")):
        raise ValueError("Run này đã được đánh giá trên test; không chạy lại để giữ giao thức một lượt.")
    device = select_device(device_name)
    manifest = load_manifest() if manifest is None else manifest
    dataset = InteractionImageDataset(manifest, "test", make_transform())
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), shuffle=False,
                        num_workers=int(config.get("num_workers", 0)), pin_memory=False)
    model = build_model(config["backbone"], pretrained=False)
    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    train_labels = manifest.loc[manifest["split"] == "train", "label_id"]
    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES))
    weights = len(train_labels) / (len(CLASS_NAMES) * counts)
    criterion = torch.nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device))
    loss, metrics, y_true, y_pred, _ = run_epoch(model, loader, criterion, device)
    metrics["loss"] = loss
    report_path = run_dir / "report.txt"
    existing = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    existing["test"] = metrics
    write_json(metrics_path, existing)
    previous = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    report_path.write_text(previous.rstrip() + "\n\n" + format_report("test", metrics, y_true, y_pred),
                           encoding="utf-8")
    log_run(run_dir, f"Đánh giá test hoàn tất: macro_f1={metrics['f1_macro']:.6f}, "
                     f"accuracy={metrics['accuracy']:.6f}; metrics đã lưu vào {metrics_path}")
    summary_path = update_results_summary()
    print(f"Đã cập nhật tổng hợp kết quả: {summary_path}")
    print(f"Test macro-F1={metrics['f1_macro']:.4f}; accuracy={metrics['accuracy']:.4f}; device={device}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Thư mục run chứa config.json và best.pt")
    parser.add_argument("--lock-run", action="store_true",
                        help="Khóa cấu hình sau khi đã xem kết quả train/val; không chạy test trong lệnh này")
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    config_path = args.run / "config.json"
    if not config_path.is_file() or not (args.run / "best.pt").is_file():
        raise FileNotFoundError("Run cần có config.json và best.pt.")
    if args.lock_run:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["locked"] = True
        write_json(config_path, config)
        update_results_summary()
        print(f"Đã khóa cấu hình {args.run}. Chạy lại evaluate.py không có --lock-run để đánh giá test.")
        return
    evaluate_run(args.run, args.device)


if __name__ == "__main__":
    main()

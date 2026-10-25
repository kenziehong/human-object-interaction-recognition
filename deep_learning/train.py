"""Train one experiment or all six baselines using train/validation only."""

from __future__ import annotations

import argparse
import csv
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from config import (ROOT, ExperimentConfig, add_experiment_arguments, config_from_args,
                    seed_everything, select_device, update_results_summary, write_json)
from data import InteractionImageDataset, CSV_PATH, IMAGE_ROOT, load_manifest, make_transform
from metrics import classification_metrics, format_report
from model import build_model, set_backbone_trainable


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()


def log_run(run_dir: Path, message: str) -> None:
    """Append a timestamped progress line and flush it immediately to disk."""
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    with (run_dir / "training.txt").open("a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    if training:
        backbone_parameters = [parameter for name, parameter in model.named_parameters()
                               if not name.startswith(("fc.", "classifier."))]
        if backbone_parameters and not any(parameter.requires_grad for parameter in backbone_parameters):
            feature_module = model if hasattr(model, "layer1") else model.features
            feature_module.eval()
            if hasattr(model, "fc"):
                model.fc.train()
            else:
                model.classifier.train()
    total_loss, sample_count, truths, predictions = 0.0, 0, [], []
    synchronize(device)
    started = time.perf_counter()
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu()) * labels.shape[0]
        sample_count += labels.shape[0]
        truths.extend(labels.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())
    synchronize(device)
    elapsed = time.perf_counter() - started
    metrics = classification_metrics(truths, predictions)
    return total_loss / sample_count, metrics, truths, predictions, elapsed


def make_optimizer(model, cfg: ExperimentConfig):
    backbone_params = [p for name, p in model.named_parameters()
                       if p.requires_grad and not name.startswith(("fc.", "classifier."))]
    classifier_params = [p for name, p in model.named_parameters()
                         if p.requires_grad and name.startswith(("fc.", "classifier."))]
    groups = [{"params": classifier_params, "lr": cfg.classifier_lr}]
    if backbone_params:
        groups.append({"params": backbone_params, "lr": cfg.backbone_lr})
    return torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)


def _new_run_dir(cfg: ExperimentConfig) -> Path:
    base = ROOT / "deep_learning/outputs" / f"{cfg.backbone}_{cfg.setup}_seed{cfg.seed}"
    candidate, index = base, 1
    while candidate.exists():
        candidate = base.with_name(f"{base.name}_{index:02d}")
        index += 1
    candidate.mkdir(parents=True)
    return candidate


def train_one(cfg: ExperimentConfig, manifest, run_dir: Path | None = None) -> Path:
    seed_everything(cfg.seed)
    device = select_device(cfg.device)
    run_dir = run_dir or _new_run_dir(cfg)
    log_run(run_dir, f"Bắt đầu run {run_dir.name}")
    log_run(run_dir, "Cấu hình: " + ", ".join(f"{key}={value}" for key, value in cfg.to_dict().items()))
    augment = cfg.setup in ("th2", "th3")
    datasets = {
        "train": InteractionImageDataset(manifest, "train", make_transform(augment)),
        "train_eval": InteractionImageDataset(manifest, "train", make_transform()),
        "val": InteractionImageDataset(manifest, "val", make_transform()),
    }
    loaders = {split: DataLoader(ds, batch_size=cfg.batch_size, shuffle=split == "train",
                                 num_workers=cfg.num_workers, pin_memory=False)
               for split, ds in datasets.items()}
    train_counts = np.bincount(manifest.loc[manifest.split == "train", "label_id"], minlength=6)
    weights = len(manifest.loc[manifest.split == "train"]) / (len(train_counts) * train_counts)
    model = build_model(cfg.backbone, cfg.pretrained).to(device)
    print(f"Runtime: torch={torch.__version__}, device={device}, model={next(model.parameters()).device}, "
          f"input={device}; MPS built={torch.backends.mps.is_built()}, "
          f"available={torch.backends.mps.is_available()}")
    log_run(run_dir, f"Runtime: torch={torch.__version__}, torchvision={__import__('torchvision').__version__}, "
                     f"device={device}, model_device={next(model.parameters()).device}, "
                     f"MPS built={torch.backends.mps.is_built()}, available={torch.backends.mps.is_available()}")
    set_backbone_trainable(model, cfg.backbone, cfg.setup == "th3" and cfg.unfreeze_epoch <= 1)
    optimizer = make_optimizer(model, cfg)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    config = cfg.to_dict()
    config.update({"device_actual": str(device), "csv_path": str(CSV_PATH), "image_root": str(IMAGE_ROOT),
                   "best_epoch": None, "locked": False,
                   "runtime": {"torch": torch.__version__, "torchvision": __import__("torchvision").__version__,
                               "machine": platform.machine(), "mps_built": torch.backends.mps.is_built(),
                               "mps_available": torch.backends.mps.is_available()}})
    write_json(run_dir / "config.json", config)
    manifest.loc[:, [column for column in ("dataset", "label", "label_id", "crop_path", "source_image", "split")
                     if column in manifest.columns]].to_csv(run_dir / "manifest.csv", index=False)
    history_path = run_dir / "history.csv"
    best_score, best_loss, stale, best_epoch = -1.0, float("inf"), 0, 0
    with history_path.open("w", newline="", encoding="utf-8") as history_file:
        writer = None
        for epoch in range(1, cfg.max_epochs + 1):
            if cfg.setup == "th3" and epoch == cfg.unfreeze_epoch:
                set_backbone_trainable(model, cfg.backbone, True)
                optimizer = make_optimizer(model, cfg)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=2)
            _, _, _, _, train_time = run_epoch(model, loaders["train"], criterion, device, optimizer)
            # Report train metrics on the complete unaugmented split in eval mode.
            train_loss, train_metrics, train_y, train_pred, train_eval_time = run_epoch(
                model, loaders["train_eval"], criterion, device)
            val_loss, val_metrics, val_y, val_pred, val_time = run_epoch(
                model, loaders["val"], criterion, device)
            scheduler.step(val_loss)
            score = val_metrics["f1_macro"]
            if score > best_score + cfg.min_delta or (abs(score - best_score) <= cfg.min_delta and val_loss < best_loss):
                best_score, best_loss, stale, best_epoch = score, val_loss, 0, epoch
                config["best_epoch"] = epoch
                torch.save({"state_dict": model.state_dict(), "backbone": cfg.backbone,
                            "class_names": list(__import__("config").CLASS_NAMES),
                            "model_name": run_dir.name, "config": config,
                            "preprocessing": {"image_size": 224, "mean": (0.485, 0.456, 0.406),
                                              "std": (0.229, 0.224, 0.225)}}, run_dir / "best.pt")
                final_metrics = {"train": {**train_metrics, "loss": train_loss},
                                 "val": {**val_metrics, "loss": val_loss}}
                reports = [format_report("train", train_metrics, train_y, train_pred),
                           format_report("val", val_metrics, val_y, val_pred)]
                write_json(run_dir / "metrics.json", final_metrics)
                (run_dir / "report.txt").write_text("\n\n".join(reports), encoding="utf-8")
                log_run(run_dir, f"Epoch {epoch}: checkpoint tốt nhất mới; val_macro_f1={score:.6f}, "
                                 f"val_loss={val_loss:.6f}; đã lưu {run_dir / 'best.pt'}")
            else:
                stale += 1
            values = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                      **{f"train_{key}": value for key, value in train_metrics.items() if key != "per_class" and key != "confusion_matrix"},
                      **{f"val_{key}": value for key, value in val_metrics.items() if key != "per_class" and key != "confusion_matrix"},
                      "learning_rate": optimizer.param_groups[0]["lr"],
                      "epoch_seconds": train_time + train_eval_time + val_time,
                      "images_per_second": len(datasets["train"]) / max(train_time, 1e-9)}
            if writer is None:
                writer = csv.DictWriter(history_file, fieldnames=list(values))
                writer.writeheader()
            writer.writerow(values)
            history_file.flush()
            print(f"{run_dir.name} epoch {epoch}/{cfg.max_epochs}: train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_macro_f1={score:.4f} ({device})")
            log_run(run_dir, f"Epoch {epoch}/{cfg.max_epochs}: train_loss={train_loss:.6f}, "
                             f"train_macro_f1={train_metrics['f1_macro']:.6f}, "
                             f"val_loss={val_loss:.6f}, val_macro_f1={score:.6f}, "
                             f"val_accuracy={val_metrics['accuracy']:.6f}, "
                             f"learning_rates={[group['lr'] for group in optimizer.param_groups]}, "
                             f"seconds={train_time + train_eval_time + val_time:.2f}, "
                             f"images_per_second={len(datasets['train']) / max(train_time, 1e-9):.2f}")
            config["last_epoch"] = epoch
            config["latest_metrics"] = {
                "train": {**train_metrics, "loss": train_loss},
                "val": {**val_metrics, "loss": val_loss},
            }
            write_json(run_dir / "config.json", config)
            update_results_summary()
            if stale >= cfg.patience:
                break
    config["best_epoch"] = best_epoch
    config["training_complete"] = True
    write_json(run_dir / "config.json", config)
    log_run(run_dir, f"Huấn luyện hoàn tất; best_epoch={best_epoch}, best_val_macro_f1={best_score:.6f}")
    summary_path = update_results_summary()
    print(f"Đã cập nhật tổng hợp kết quả: {summary_path}")
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_experiment_arguments(parser)
    parser.add_argument("--all-baselines", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest()
    if args.all_baselines:
        for backbone in ("resnet50", "efficientnet_b0"):
            for setup in ("th1", "th2", "th3"):
                values = vars(args).copy()
                values.update(backbone=backbone, setup=setup)
                current = argparse.Namespace(**values)
                cfg = config_from_args(current)
                train_one(cfg, manifest)
    else:
        train_one(config_from_args(args), manifest)


if __name__ == "__main__":
    main()

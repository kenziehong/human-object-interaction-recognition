"""Randomly sample hyperparameters and rank runs by validation macro-F1."""

from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

from config import ExperimentConfig, ROOT, update_results_summary, write_json
from data import load_manifest
from evaluate import evaluate_run
from train import train_one


BACKBONES = ("resnet50", "efficientnet_b0")
SETUPS = ("th1", "th2", "th3")
CLASSIFIER_LRS = (3e-4, 1e-3, 3e-3)
TH3_CLASSIFIER_LRS = (1e-4, 3e-4, 1e-3)
BACKBONE_LRS = (1e-5, 3e-5, 1e-4)
WEIGHT_DECAYS = (1e-5, 1e-4, 1e-3)
BATCH_SIZES = (16, 32)
UNFREEZE_EPOCH = 3


def candidate_configs(backbone: str, setup: str, device: str, num_workers: int):
    """Enumerate the supported search space before random sampling."""
    classifier_lrs = TH3_CLASSIFIER_LRS if setup == "th3" else CLASSIFIER_LRS
    backbone_lrs = BACKBONE_LRS if setup == "th3" else (1e-5,)
    for classifier_lr, backbone_lr, weight_decay, batch_size in itertools.product(
            classifier_lrs, backbone_lrs, WEIGHT_DECAYS, BATCH_SIZES):
        yield ExperimentConfig(
            backbone=backbone,
            setup=setup,
            seed=42,
            batch_size=batch_size,
            max_epochs=30,
            patience=6,
            min_delta=0.0,
            classifier_lr=classifier_lr,
            backbone_lr=backbone_lr,
            weight_decay=weight_decay,
            unfreeze_epoch=UNFREEZE_EPOCH,
            num_workers=num_workers,
            device=device,
            pretrained=True,
        )


def candidate_rank(candidate: dict) -> tuple:
    val = candidate["val"]
    return val.get("f1_macro", -1.0), -val.get("loss", float("inf"))


def select_winners(candidates: list[dict]) -> dict:
    winners = {}
    for item in candidates:
        key = (item["config"]["backbone"], item["config"]["setup"])
        if key not in winners or candidate_rank(item) > candidate_rank(winners[key]):
            winners[key] = item
    return winners


def run_candidate(cfg: ExperimentConfig, manifest) -> dict:
    run_dir = train_one(cfg, manifest)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    return {"config": cfg.to_dict(), "run_dir": run_dir,
            "metrics": metrics, "val": metrics["val"]}


def metric_lines(split_name: str, metrics: dict) -> list[str]:
    lines = [
        (f"{split_name}: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}, "
         f"macro P/R/F1={metrics['precision_macro']:.4f}/"
         f"{metrics['recall_macro']:.4f}/{metrics['f1_macro']:.4f}, "
         f"weighted P/R/F1={metrics['precision_weighted']:.4f}/"
         f"{metrics['recall_weighted']:.4f}/{metrics['f1_weighted']:.4f}, "
         f"balanced_accuracy={metrics['balanced_accuracy']:.4f}"),
        "  Theo lớp (precision / recall / F1 / support):",
    ]
    for class_name, values in metrics["per_class"].items():
        lines.append(
            f"  {class_name}: {values['precision']:.4f} / {values['recall']:.4f} / "
            f"{values['f1-score']:.4f} / {values['support']}")
    return lines


def new_search_report_path() -> Path:
    outputs_dir = ROOT / "deep_learning/outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    report_path = outputs_dir / "random_search_summary.txt"
    suffix = 1
    while report_path.exists():
        report_path = outputs_dir / f"random_search_summary_{suffix:02d}.txt"
        suffix += 1
    return report_path


def write_search_report(candidates: list[dict], winners: dict, search_seed: int,
                        report_path: Path, status: str) -> None:
    lines = [
        "TỔNG KẾT RANDOM SEARCH DEEP LEARNING",
        f"Seed lấy mẫu: {search_seed}; các run dùng seed huấn luyện: 42.",
        "Xếp hạng theo val macro-F1; nếu hòa, ưu tiên val loss thấp hơn.",
        "Test chỉ được đánh giá sau khi chọn cấu hình thắng; test không dùng để xếp hạng.",
        f"Trạng thái: {status}",
        "",
    ]
    for key in sorted(winners):
        winner = winners[key]
        cfg, val = winner["config"], winner["val"]
        lines.extend([
            "=" * 84,
            f"TỐT NHẤT: {key[0]} | {key[1].upper()}",
            (f"Tham số: batch_size={cfg['batch_size']}, classifier_lr={cfg['classifier_lr']}, "
             f"backbone_lr={cfg['backbone_lr']}, weight_decay={cfg['weight_decay']}, "
             f"unfreeze_epoch={cfg['unfreeze_epoch']}, epochs={cfg['max_epochs']}, "
             f"patience={cfg['patience']}"),
        ])
        lines.extend(metric_lines("Validation", val))
        if "test" in winner:
            lines.extend(metric_lines("Test", winner["test"]))
        else:
            lines.append("Test: đang chờ random search hoàn tất và run thắng được đánh giá.")
        lines.extend([
            f"Checkpoint: {winner['run_dir'] / 'best.pt'}",
            f"Chi tiết đầy đủ: {winner['run_dir'] / 'metrics.json'} | {winner['run_dir'] / 'report.txt'}",
            "",
        ])

    lines.extend(["=" * 84, "TẤT CẢ CẤU HÌNH ĐÃ CHẠY"])
    for key in sorted(winners):
        group = sorted((item for item in candidates
                        if (item["config"]["backbone"], item["config"]["setup"]) == key),
                       key=candidate_rank, reverse=True)
        lines.append(f"\n{key[0]} | {key[1].upper()} ({len(group)} cấu hình)")
        for rank, item in enumerate(group, 1):
            cfg, val = item["config"], item["val"]
            lines.append(
                f"{rank:>2}. macro-F1={val['f1_macro']:.4f}, val_loss={val['loss']:.4f}, "
                f"batch={cfg['batch_size']}, classifier_lr={cfg['classifier_lr']}, "
                f"backbone_lr={cfg['backbone_lr']}, weight_decay={cfg['weight_decay']}, "
                f"checkpoint={item['run_dir'] / 'best.pt'}")

    if not candidates:
        lines.append("Chưa có cấu hình nào hoàn tất.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("mps", "cpu", "auto"), default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--configs-per-backbone", type=int, default=10,
                        help="Số cấu hình tối đa cho mỗi cặp TH/backbone (1–10).")
    parser.add_argument("--search-seed", type=int, default=42,
                        help="Seed để lấy mẫu ngẫu nhiên cấu hình; không phải seed huấn luyện.")
    parser.add_argument("--backbones", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    parser.add_argument("--setups", nargs="+", choices=SETUPS, default=list(SETUPS))
    args = parser.parse_args()
    if not 1 <= args.configs_per_backbone <= 10:
        parser.error("--configs-per-backbone phải nằm trong khoảng 1–10")

    report_path = new_search_report_path()
    write_search_report([], {}, args.search_seed, report_path, "Đang chuẩn bị dữ liệu.")
    manifest = load_manifest()
    rng = random.Random(args.search_seed)
    candidates = []
    for setup in args.setups:
        for backbone in args.backbones:
            search_space = list(candidate_configs(backbone, setup, args.device, args.num_workers))
            sampled = rng.sample(search_space, min(args.configs_per_backbone, len(search_space)))
            print(f"Random search {backbone}/{setup}: {len(sampled)} cấu hình")
            for run_index, cfg in enumerate(sampled, 1):
                print(f"[{run_index}/{len(sampled)}] {backbone}/{setup}: "
                      f"batch={cfg.batch_size}, classifier_lr={cfg.classifier_lr}, "
                      f"backbone_lr={cfg.backbone_lr}, weight_decay={cfg.weight_decay}")
                write_search_report(candidates, select_winners(candidates), args.search_seed, report_path,
                    f"Đang chạy {backbone}/{setup}, cấu hình {run_index}/{len(sampled)}.")
                candidates.append(run_candidate(cfg, manifest))
                write_search_report(candidates, select_winners(candidates), args.search_seed, report_path,
                                    f"Hoàn tất {backbone}/{setup}, cấu hình {run_index}/{len(sampled)}.")

    winners = select_winners(candidates)

    # Test is kept out of the search and run once only for each selected winner.
    for key, winner in winners.items():
        run_dir = winner["run_dir"]
        config_path = run_dir / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config["locked"] = True
        write_json(config_path, config)
        print(f"Đánh giá test cho cấu hình thắng {key[0]}/{key[1]}: {run_dir.name}")
        winner["test"] = evaluate_run(run_dir, args.device, manifest)
        write_search_report(candidates, winners, args.search_seed, report_path,
                            f"Đã đánh giá test cho {key[0]}/{key[1]}.")

    write_search_report(candidates, winners, args.search_seed, report_path, "Hoàn tất.")
    summary_path = update_results_summary()
    print(f"Đã hoàn tất random search. Báo cáo: {report_path}")
    print(f"Tổng hợp tất cả run: {summary_path}")


if __name__ == "__main__":
    main()

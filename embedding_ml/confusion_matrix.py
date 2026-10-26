#!/usr/bin/env python3
"""Evaluate a saved embedding classifier on the test split and save its confusion matrix."""

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

try:  # Support both `python embedding_ml/confusion_matrix.py` and package imports.
    from .common import load_split, read_embedding_data
except ImportError:  # pragma: no cover - direct script entry point.
    from common import load_split, read_embedding_data


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=PROJECT_ROOT / "embedding_ml/outputs/models_v3/resnet50_xgboost.joblib",
        help="File model .joblib đã huấn luyện.",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "data/embedding_csv/embeddings_v3.csv",
        help="CSV chứa embedding và split train/val/test.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "embedding_ml/outputs/confusion_matrix_resnet50_xgboost_test.png",
        help="Đường dẫn ảnh ma trận nhầm lẫn đầu ra.",
    )
    parser.add_argument(
        "--normalize",
        choices=("true", "pred", "all"),
        help="Chuẩn hóa ma trận theo nhãn thật, nhãn dự đoán hoặc toàn bộ mẫu.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {args.model}")

    bundle = joblib.load(args.model)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("File model không có cấu trúc bundle tương thích.")
    embedding = bundle.get("embedding")
    if not isinstance(embedding, str) or not embedding:
        raise ValueError("Bundle model thiếu metadata 'embedding'.")
    estimator = bundle["model"]

    frame, csv_label_names = read_embedding_data(args.csv)
    X_test, y_test = load_split(frame, "test", embedding)
    y_pred = estimator.predict(X_test)

    label_names = bundle.get("label_names") or csv_label_names
    labels = sorted(set(y_test.tolist()) | set(y_pred.tolist()))
    tick_labels = [
        str(label_names.get(label, label_names.get(str(label), str(label))))
        for label in labels
    ]
    matrix = confusion_matrix(
        y_test, y_pred, labels=labels, normalize=args.normalize
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig_width = max(8, len(labels) * 1.35)
    fig, ax = plt.subplots(figsize=(fig_width, fig_width * 0.82))
    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=tick_labels)
    display.plot(
        ax=ax,
        cmap="Blues",
        values_format=".2f" if args.normalize else "d",
        colorbar=True,
        xticks_rotation=35,
    )
    ax.set_title(f"Confusion matrix: {args.model.stem} (test, n={len(y_test):,})")
    fig.tight_layout()
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)

    report = classification_report(
        y_test,
        y_pred,
        labels=labels,
        target_names=tick_labels,
        digits=4,
        zero_division=0,
    )
    print(f"Model: {args.model}")
    print(f"Embedding: {embedding}")
    print(f"Test samples: {len(y_test):,}")
    print(f"Confusion matrix image: {args.output}")
    print("\nClassification report (test):")
    print(report)
    print("Confusion matrix (rows=true, columns=predicted):")
    print(pd.DataFrame(matrix, index=tick_labels, columns=tick_labels).to_string())


if __name__ == "__main__":
    main()

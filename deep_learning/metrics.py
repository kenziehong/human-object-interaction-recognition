"""Consistent classification metrics and human-readable reports."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)

from config import CLASS_NAMES


def classification_metrics(y_true, y_pred) -> dict:
    labels = list(range(len(CLASS_NAMES)))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                            average="macro", zero_division=0)
    weighted = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                               average="weighted", zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(macro[0]), "recall_macro": float(macro[1]), "f1_macro": float(macro[2]),
        "precision_weighted": float(weighted[0]), "recall_weighted": float(weighted[1]),
        "f1_weighted": float(weighted[2]), "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "per_class": {name: {"precision": float(precision[i]), "recall": float(recall[i]),
                             "f1-score": float(f1[i]), "support": int(support[i])}
                      for i, name in enumerate(CLASS_NAMES)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def format_report(split: str, metrics: dict, y_true, y_pred) -> str:
    labels = list(range(len(CLASS_NAMES)))
    return (f"[{split}]\nclass_order={list(CLASS_NAMES)}\n"
            f"accuracy={metrics['accuracy']:.6f}, precision_macro={metrics['precision_macro']:.6f}, "
            f"recall_macro={metrics['recall_macro']:.6f}, f1_macro={metrics['f1_macro']:.6f}, "
            f"precision_weighted={metrics['precision_weighted']:.6f}, "
            f"recall_weighted={metrics['recall_weighted']:.6f}, "
            f"f1_weighted={metrics['f1_weighted']:.6f}, "
            f"balanced_accuracy={metrics['balanced_accuracy']:.6f}\n\n"
            + classification_report(y_true, y_pred, labels=labels, target_names=CLASS_NAMES,
                                    digits=4, zero_division=0)
            + "Confusion matrix (rows=true, columns=predicted):\n"
            + np.array2string(np.asarray(metrics["confusion_matrix"]), separator=", "))


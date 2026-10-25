"""Shared data, model-search, and metric helpers for embedding experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

EMBEDDINGS = ("resnet50", "efficientnet_b0")
CLASSIFIERS = ("logreg", "svm", "random_forest", "xgboost")
SPLITS = ("train", "val", "test")


def read_embedding_data(csv_path: Path) -> tuple[pd.DataFrame, dict[int, str]]:
    """Read the combined embedding CSV and validate its split/label columns."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV embedding: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"dataset", "label_id"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"CSV thiếu cột bắt buộc: {', '.join(sorted(missing))}")
    invalid_splits = set(frame["dataset"].dropna().unique()) - set(SPLITS)
    if invalid_splits:
        raise ValueError(f"Split không hợp lệ: {sorted(invalid_splits)}")
    labels = {}
    if "label" in frame:
        labels = (frame.drop_duplicates("label_id").set_index("label_id")["label"].astype(str).to_dict())
    return frame, {int(key): value for key, value in labels.items()}


def load_split(frame: pd.DataFrame, split: str, embedding: str) -> tuple[np.ndarray, np.ndarray]:
    """Decode one split's JSON vectors into a feature matrix and integer labels."""
    column = f"{embedding}_embedding"
    if split not in SPLITS or column not in frame:
        raise ValueError(f"CSV thiếu split '{split}' hoặc embedding '{column}'.")
    rows = frame.loc[frame["dataset"] == split]
    if rows.empty:
        raise ValueError(f"Không có dữ liệu cho split '{split}'.")
    if rows[column].isna().any():
        raise ValueError(f"Embedding '{column}' có giá trị rỗng trong split '{split}'.")
    try:
        X = np.asarray([json.loads(value) for value in rows[column]], dtype=np.float32)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không đọc được vector trong cột '{column}'.") from exc
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError(f"Embedding '{column}' phải là ma trận số hữu hạn.")
    return X, rows["label_id"].to_numpy(dtype=int)


def calculate_metrics(y_true, y_pred, labels=None) -> dict[str, float]:
    """Return accuracy and macro/weighted precision, recall, and F1."""
    labels = labels if labels is not None else sorted(set(y_true) | set(y_pred))
    metrics = {"accuracy": float(accuracy_score(y_true, y_pred))}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=average, zero_division=0
        )
        metrics.update({f"precision_{average}": float(precision),
                        f"recall_{average}": float(recall), f"f1_{average}": float(f1)})
    return metrics


def make_search(classifier: str, seed: int):
    """Return an estimator pipeline and randomized-search parameter distributions."""
    if classifier == "logreg":
        estimator = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("model", estimator)]), {
            # All three embeddings favored small C values (~7e-4 to 1.9e-3).
            "model__C": loguniform(1e-5, 1e-2),
        }
    if classifier == "svm":
        estimator = SVC(class_weight="balanced", random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("model", estimator)]), [
            {"model__kernel": ["linear"], "model__C": loguniform(1e-3, 1e2)},
            # Best RBF runs used C ~= 5-8 and gamma ~= 4e-4-1.6e-3.
            {"model__kernel": ["rbf"], "model__C": loguniform(1.0, 30.0),
             "model__gamma": loguniform(1e-4, 5e-3)},
        ]
    if classifier == "random_forest":
        # Candidate search is sequential, so let each forest use all CPU cores.
        estimator = RandomForestClassifier(n_jobs=-1, class_weight="balanced", random_state=seed)
        return Pipeline([("model", estimator)]), {
            "model__n_estimators": [100, 200, 400],
            "model__max_depth": [None, 20, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.5],
            "model__class_weight": ["balanced"],
        }
    if classifier == "xgboost":
        # Import the native XGBoost runtime only when this estimator is needed.
        # Importing it for unrelated tasks such as SVM inference loads a second
        # OpenMP runtime alongside PyTorch on macOS and can crash the process.
        from xgboost import XGBClassifier

        # Candidate search is sequential, so let XGBoost use all CPU cores.
        estimator = XGBClassifier(objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1,
                                  random_state=seed, tree_method="hist")
        return Pipeline([("model", estimator)]), {
            "model__n_estimators": randint(100, 601), "model__max_depth": randint(2, 8),
            "model__learning_rate": loguniform(0.01, 0.2),
            "model__subsample": uniform(0.7, 0.3), "model__colsample_bytree": uniform(0.7, 0.3),
            # Focus on a compact range to reduce training time per candidate.
            "model__min_child_weight": loguniform(0.5, 10.0), "model__gamma": uniform(0.0, 0.5),
            "model__reg_alpha": loguniform(1e-6, 1e-1), "model__reg_lambda": loguniform(1e-2, 30.0),
        }
    raise ValueError(f"Classifier không được hỗ trợ: {classifier}")

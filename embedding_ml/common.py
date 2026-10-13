"""Shared data loading and classifier definitions for embedding experiments."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier


EMBEDDINGS = ("resnet50", "efficientnet_b0", "vit_b_16")
CLASSIFIERS = ("logreg", "svm", "random_forest", "xgboost")


def load_split(csv_path: Path, split: str, embedding: str):
    """Load one split as a feature matrix and integer labels."""
    frame = pd.read_csv(csv_path)
    column = f"{embedding}_embedding"
    if column not in frame or "dataset" not in frame or "label_id" not in frame:
        raise ValueError(f"CSV thiếu cột cần thiết cho embedding '{embedding}'.")

    frame = frame.loc[frame["dataset"] == split]
    if frame.empty:
        raise ValueError(f"Không có dữ liệu cho split '{split}'.")
    if frame[column].isna().any():
        raise ValueError(f"Embedding '{column}' có giá trị rỗng trong split '{split}'.")

    try:
        X = np.asarray([json.loads(value) for value in frame[column]], dtype=np.float32)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Không đọc được vector trong cột '{column}'.") from exc
    if X.ndim != 2 or not np.isfinite(X).all():
        raise ValueError(f"Embedding '{column}' phải là ma trận số hữu hạn.")
    return X, frame["label_id"].to_numpy(dtype=int)


def make_search(classifier: str, seed: int):
    """Return a pipeline and its randomized-search parameter distributions."""
    if classifier == "logreg":
        estimator = LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("model", estimator)]), {
            "model__C": np.logspace(-3, 3, 30),
        }
    if classifier == "svm":
        estimator = SVC(class_weight="balanced", random_state=seed)
        return Pipeline([("scale", StandardScaler()), ("model", estimator)]), {
            "model__C": np.logspace(-3, 3, 30),
            "model__kernel": ["linear", "rbf"],
            "model__gamma": ["scale", "auto"],
        }
    if classifier == "random_forest":
        estimator = RandomForestClassifier(n_jobs=-1, class_weight="balanced", random_state=seed)
        return Pipeline([("model", estimator)]), {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [None, 10, 20, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", 0.5],
        }
    if classifier == "xgboost":
        estimator = XGBClassifier(
            objective="multi:softprob", eval_metric="mlogloss", n_jobs=-1,
            random_state=seed, tree_method="hist",
        )
        return Pipeline([("model", estimator)]), {
            "model__n_estimators": [200, 400, 600],
            "model__max_depth": [3, 5, 7, 9],
            "model__learning_rate": [0.01, 0.03, 0.1, 0.2],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        }
    raise ValueError(f"Classifier không được hỗ trợ: {classifier}")

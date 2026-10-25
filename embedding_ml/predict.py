#!/usr/bin/env python3
"""Predict one or more images with one saved embedding classifier."""

import argparse
import json
import os
from pathlib import Path

# Keep single-image inference from starting competing OpenMP worker pools on
# macOS. Set these before NumPy, scikit-learn, or PyTorch load native runtimes.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import joblib
import numpy as np
import torch
from PIL import Image

try:  # Support both `python embedding_ml/predict.py` and package imports.
    from .common import EMBEDDINGS
    from .extract_embedding import build_model_and_transform, choose_device, extract_batch_embeddings
except ImportError:  # pragma: no cover - exercised by the direct script entry point.
    from common import EMBEDDINGS
    from extract_embedding import build_model_and_transform, choose_device, extract_batch_embeddings


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", type=Path, help="Đường dẫn file model .joblib")
    parser.add_argument("images", nargs="+", type=Path, help="Một hoặc nhiều đường dẫn ảnh")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def extract_images(image_paths, embedding, device, batch_size):
    """Extract one backbone's feature vectors for a list of images."""
    feature_model, transform, _ = build_model_and_transform(embedding, preprocessing="weights")
    feature_model = feature_model.to(device).eval()
    batches = []
    for start in range(0, len(image_paths), batch_size):
        tensors = []
        for path in image_paths[start:start + batch_size]:
            with Image.open(path) as image:
                tensors.append(transform(image.convert("RGB")))
        images = torch.stack(tensors)
        batches.append(extract_batch_embeddings(feature_model, images, embedding, device).numpy())
    del feature_model
    if device.type == "mps" and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(batches)


def get_confidence(model, features, prediction):
    """Return predicted-class probability or a softmax score from SVM margins."""
    classes = np.asarray(model.classes_)
    class_index = int(np.flatnonzero(classes == prediction)[0])
    if hasattr(model, "predict_proba"):
        probabilities = np.asarray(model.predict_proba(features), dtype=float)
        if probabilities.ndim == 2 and probabilities.shape[1] == len(classes):
            return float(probabilities[0, class_index])

    scores = np.asarray(model.decision_function(features), dtype=float)
    if scores.ndim == 1:
        if len(classes) == 2:  # Binary SVC: positive margin corresponds to classes_[1].
            scores = np.array([[-scores[0], scores[0]]])
        else:
            scores = scores.reshape(1, -1)
    if scores.shape[1] != len(classes):
        raise ValueError(
            "decision_function output does not match the estimator's classes; "
            "cannot calculate confidence."
        )
    scores = scores[0]
    scores -= scores.max()
    probabilities = np.exp(scores) / np.exp(scores).sum()
    return float(probabilities[class_index])


def predict(model_path, image_paths, device="auto", batch_size=16) -> str:
    """Predict one image or a list and return ordered JSON Lines with image paths."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy model: {model_path}")
    image_paths = [Path(path) for path in ([image_paths] if isinstance(image_paths, (str, Path)) else image_paths)]
    if not image_paths:
        raise ValueError("Cần truyền ít nhất một đường dẫn ảnh.")
    for image_path in image_paths:
        if not image_path.is_file():
            raise FileNotFoundError(f"Không tìm thấy ảnh: {image_path}")
    if batch_size < 1:
        raise ValueError("batch_size phải lớn hơn 0.")

    bundle = joblib.load(model_path)
    embedding = bundle.get("embedding")
    classifier = bundle.get("classifier")
    if embedding not in EMBEDDINGS or "model" not in bundle:
        raise ValueError("File model không có metadata embedding/classifier tương thích.")
    estimator = bundle["model"]
    vectors = extract_images(image_paths, embedding, choose_device(device), batch_size)
    predictions = estimator.predict(vectors)
    label_names = bundle.get("label_names", {})
    model_name = f"{embedding}_{classifier}" if classifier else model_path.stem

    records = []
    for index, prediction_value in enumerate(predictions):
        prediction = int(prediction_value)
        label = label_names.get(prediction, label_names.get(str(prediction), str(prediction)))
        confidence = get_confidence(estimator, vectors[index:index + 1], prediction)
        records.append({"image_path": str(image_paths[index]), "label_id": prediction,
                        "label": label, "confidence": confidence, "model_name": model_name})
    return "\n".join(json.dumps(record, ensure_ascii=False) for record in records)


def main():
    args = parse_args()
    print(predict(args.model_path, args.images, args.device, args.batch_size))


if __name__ == "__main__":
    main()

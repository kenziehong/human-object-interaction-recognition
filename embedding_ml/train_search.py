#!/usr/bin/env python3
"""Tune embedding classifiers on train/val, evaluate on test, and save each best model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.metrics import classification_report
from sklearn.model_selection import ParameterSampler

from common import (CLASSIFIERS, EMBEDDINGS, calculate_metrics, load_split,
                    make_search, read_embedding_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/embedding_csv/embeddings_v3.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("embedding_ml/outputs/models_v3"))
    parser.add_argument("--results-file", type=Path, default=Path("embedding_ml/outputs/all_results_v3.txt"))
    parser.add_argument("--n-iter", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def search_best_params(classifier, seed, n_iter, X_train, y_train, X_val, y_val):
    """Fit each sampled configuration once and keep its validation winner."""
    pipeline, params = make_search(classifier, seed)
    best_model, best_params, best_score = None, None, float("-inf")
    for index, candidate in enumerate(ParameterSampler(params, n_iter=n_iter, random_state=seed), 1):
        model = clone(pipeline).set_params(**candidate)
        model.fit(X_train, y_train)
        score = calculate_metrics(y_val, model.predict(X_val))["f1_macro"]
        print(f"Candidate {index}/{n_iter}: val_f1_macro={score:.4f}", flush=True)
        if score > best_score:
            best_model, best_params, best_score = model, candidate, score
    return best_model, best_params, best_score


def evaluate_model(model, X, y, labels, label_names):
    prediction = model.predict(X)
    metrics = calculate_metrics(y, prediction, labels)
    report = classification_report(y, prediction, labels=labels,
                                   target_names=[label_names.get(label, str(label)) for label in labels],
                                   digits=4, zero_division=0)
    return metrics, report


def metric_text(metrics):
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())


def run_experiment(embedding, classifier, args, frame, label_names, model_dir, splits):
    """Search on train, then report and save the validation winner."""
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    best_model, best_params, search_score = search_best_params(
        classifier, args.seed, args.n_iter, X_train, y_train, X_val, y_val
    )
    labels = sorted(label_names) if label_names else sorted(set(y_train) | set(y_val))

    X_test, y_test = splits["test"]
    train_metrics, train_report = evaluate_model(best_model, X_train, y_train, labels, label_names)
    val_metrics, val_report = evaluate_model(best_model, X_val, y_val, labels, label_names)
    test_metrics, test_report = evaluate_model(best_model, X_test, y_test, labels, label_names)

    model_path = model_dir / f"{embedding}_{classifier}.joblib"
    joblib.dump({"model": best_model, "embedding": embedding, "classifier": classifier,
                 "label_names": label_names, "best_params": best_params,
                 "selection_metric": "validation_f1_macro", "validation_f1_macro": val_metrics["f1_macro"]},
                model_path)
    details = (f"\n{'=' * 78}\n{embedding} + {classifier}\n"
               f"Best params: {json.dumps(best_params, sort_keys=True)}\n"
               f"Search validation macro-F1: {search_score:.4f}\n"
               f"[train, model fit on train] {metric_text(train_metrics)}\n{train_report}\n"
               f"[val, model fit on train] {metric_text(val_metrics)}\n{val_report}\n"
               f"[test, saved model fit on train] {metric_text(test_metrics)}\n{test_report}\n"
               f"Saved model: {model_path}\n")
    print(details, flush=True)
    return details


def main():
    args = parse_args()
    args.model_dir.mkdir(parents=True, exist_ok=True)
    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    frame, label_names = read_embedding_data(args.csv)
    split_cache = {
        embedding: {split: load_split(frame, split, embedding) for split in ("train", "val", "test")}
        for embedding in EMBEDDINGS
    }

    report_path = args.results_file
    report = report_path.read_text(encoding="utf-8") if report_path.exists() else f"Embedding classifier search\nInput: {args.csv}\n"
    for embedding in EMBEDDINGS:
        for classifier in CLASSIFIERS:
            model_path = args.model_dir / f"{embedding}_{classifier}.joblib"
            if model_path.exists():
                print(f"Skipping {embedding} + {classifier}: {model_path} already exists", flush=True)
                continue
            report += run_experiment(embedding, classifier, args, frame, label_names,
                                     args.model_dir, split_cache[embedding])
            report_path.write_text(report, encoding="utf-8")
    print(f"Models: {args.model_dir}")


if __name__ == "__main__":
    main()

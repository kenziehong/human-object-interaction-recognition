#!/usr/bin/env python3
"""Tune embedding classifiers on train/val, evaluate on test, and save each best model."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from common import (CLASSIFIERS, EMBEDDINGS, calculate_metrics, load_split,
                    make_search, read_embedding_data)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/embedding_csv/embeddings_v2.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("embedding_ml/outputs"))
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def search_best_params(classifier, seed, n_iter, X_train, y_train, X_val, y_val):
    """Tune on train and select parameters by validation macro-F1."""
    pipeline, params = make_search(classifier, seed)
    X_search = np.concatenate([X_train, X_val])
    y_search = np.concatenate([y_train, y_val])
    split = PredefinedSplit(np.r_[np.full(len(y_train), -1), np.zeros(len(y_val))])
    search = RandomizedSearchCV(pipeline, params, n_iter=n_iter, scoring="f1_macro", cv=split,
                                n_jobs=-1, random_state=seed, refit=False, verbose=1)
    search.fit(X_search, y_search)
    return search.best_params_, float(search.best_score_)


def fit_model(classifier, seed, params, X, y):
    """Fit the selected configuration on the rows available for its final use."""
    model, _ = make_search(classifier, seed)
    return model.set_params(**params).fit(X, y)


def evaluate_model(model, X, y, labels, label_names):
    prediction = model.predict(X)
    metrics = calculate_metrics(y, prediction, labels)
    report = classification_report(y, prediction, labels=labels,
                                   target_names=[label_names.get(label, str(label)) for label in labels],
                                   digits=4, zero_division=0)
    return metrics, report


def metric_text(metrics):
    return ", ".join(f"{name}={value:.4f}" for name, value in metrics.items())


def run_experiment(embedding, classifier, args, frame, label_names, model_dir):
    """Search, report validation, refit on train+val, and evaluate/save test model."""
    X_train, y_train = load_split(frame, "train", embedding)
    X_val, y_val = load_split(frame, "val", embedding)
    best_params, search_score = search_best_params(
        classifier, args.seed, args.n_iter, X_train, y_train, X_val, y_val
    )
    labels = sorted(label_names) if label_names else sorted(set(y_train) | set(y_val))

    validation_model = fit_model(classifier, args.seed, best_params, X_train, y_train)
    val_metrics, val_report = evaluate_model(validation_model, X_val, y_val, labels, label_names)
    final_model = fit_model(classifier, args.seed, best_params,
                            np.concatenate([X_train, X_val]), np.concatenate([y_train, y_val]))
    # Keep the test split out of tuning and final model fitting.
    X_test, y_test = load_split(frame, "test", embedding)
    test_metrics, test_report = evaluate_model(final_model, X_test, y_test, labels, label_names)

    model_path = model_dir / f"{embedding}_{classifier}.joblib"
    joblib.dump({"model": final_model, "embedding": embedding, "classifier": classifier,
                 "label_names": label_names, "best_params": best_params,
                 "selection_metric": "validation_f1_macro", "validation_f1_macro": val_metrics["f1_macro"]},
                model_path)
    result = {"embedding": embedding, "classifier": classifier, "search_val_f1_macro": search_score,
              "best_params": json.dumps(best_params, sort_keys=True), "model_path": str(model_path)}
    result.update({f"val_{key}": value for key, value in val_metrics.items()})
    result.update({f"test_{key}": value for key, value in test_metrics.items()})
    details = (f"\n{'=' * 78}\n{embedding} + {classifier}\n"
               f"Best params: {json.dumps(best_params, sort_keys=True)}\n"
               f"Search validation macro-F1: {search_score:.4f}\n"
               f"[val, model fit on train] {metric_text(val_metrics)}\n{val_report}\n"
               f"[test, saved model fit on train+val] {metric_text(test_metrics)}\n{test_report}\n"
               f"Saved model: {model_path}\n")
    print(details, flush=True)
    return result, details


def save_results(results, output_dir, report):
    """Persist machine-readable metrics, per-class reports, and a summary text file."""
    pd.DataFrame(results).to_csv(output_dir / "metrics.csv", index=False)
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (output_dir / "all_results_v2.txt").write_text(report, encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models_v2"
    model_dir.mkdir(parents=True, exist_ok=True)
    frame, label_names = read_embedding_data(args.csv)

    results, report = [], [f"Embedding classifier search\nInput: {args.csv}\n"]
    for embedding in EMBEDDINGS:
        for classifier in CLASSIFIERS:
            result, details = run_experiment(embedding, classifier, args, frame, label_names, model_dir)
            results.append(result)
            report.append(details)
    save_results(results, args.output_dir, "\n".join(report))
    print(f"\nMetrics: {args.output_dir / 'metrics.csv'}")
    print(f"Models: {model_dir}")


if __name__ == "__main__":
    main()

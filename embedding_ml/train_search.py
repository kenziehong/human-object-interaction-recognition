#!/usr/bin/env python3
"""Search on train/val, then fit the best configuration and evaluate test."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import PredefinedSplit, RandomizedSearchCV

from common import CLASSIFIERS, EMBEDDINGS, load_split, make_search


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=Path("data/embedding_csv/splits/embeddings_split.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("embedding_ml/outputs"))
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "all_results.txt"
    report_path.write_text("Random search results\n", encoding="utf-8")
    label_frame = pd.read_csv(args.csv, usecols=["label_id", "label"])
    label_names = (
        label_frame.drop_duplicates("label_id")
        .set_index("label_id")["label"]
        .to_dict()
    )

    def report(y_true, y_pred, labels):
        return classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=[label_names.get(label, str(label)) for label in labels],
            digits=4,
            zero_division=0,
        )

    results = []
    test_summaries = []
    for embedding in EMBEDDINGS:
        X_train, y_train = load_split(args.csv, "train", embedding)
        X_val, y_val = load_split(args.csv, "val", embedding)
        X_search = np.concatenate([X_train, X_val])
        y_search = np.concatenate([y_train, y_val])
        # -1 rows are training data; fold 0 is the held-out validation set.
        validation_split = PredefinedSplit(
            np.concatenate([np.full(len(y_train), -1), np.zeros(len(y_val))])
        )
        for classifier in CLASSIFIERS:
            pipeline, params = make_search(classifier, args.seed)
            search = RandomizedSearchCV(
                pipeline, params, n_iter=args.n_iter, scoring="f1_macro",
                cv=validation_split, n_jobs=-1, random_state=args.seed,
                refit=False, verbose=1,
            )
            search.fit(X_search, y_search)

            # Fit on train only to report held-out validation performance.
            best_model = pipeline.set_params(**search.best_params_)
            best_model.fit(X_train, y_train)

            split_data = {
                "train": (X_train, y_train),
                "val": (X_val, y_val),
            }
            labels = sorted(set(y_train) | set(y_val))
            metrics = {}
            blocks = [
                f"\n{'=' * 78}\n{embedding} + {classifier}\n"
                f"Best params: {json.dumps(search.best_params_, sort_keys=True)}\n"
                f"Search best validation macro-F1: {search.best_score_:.4f}\n"
                "Saved/reporting model fit: train only\n"
            ]
            for split, (X, y) in split_data.items():
                prediction = best_model.predict(X)
                accuracy = accuracy_score(y, prediction)
                macro_f1 = f1_score(y, prediction, average="macro", zero_division=0)
                metrics[split] = (accuracy, macro_f1)
                blocks.append(
                    f"\n[{split}] accuracy: {accuracy:.4f}; macro-F1: {macro_f1:.4f}\n"
                    + report(y, prediction, labels)
                )

            result_text = "".join(blocks)
            print(result_text, flush=True)
            with report_path.open("a", encoding="utf-8") as report_file:
                report_file.write(result_text + "\n")

            result = {
                "embedding": embedding,
                "classifier": classifier,
                "val_macro_f1": metrics["val"][1],
                "val_accuracy": metrics["val"][0],
                "best_params": json.dumps(search.best_params_),
            }
            results.append(result)

            # Reuse the same train-only best model for test evaluation.
            model_path = model_dir / f"{embedding}_{classifier}.joblib"
            joblib.dump(best_model, model_path)

            X_test, y_test = load_split(args.csv, "test", embedding)
            test_prediction = best_model.predict(X_test)
            test_summary = {
                "embedding": embedding,
                "classifier": classifier,
                "test_macro_f1": f1_score(
                    y_test, test_prediction, average="macro", zero_division=0
                ),
                "test_accuracy": accuracy_score(y_test, test_prediction),
                "best_params": search.best_params_,
            }
            test_summaries.append(test_summary)
            test_labels = sorted(set(y_train) | set(y_test))
            test_report = report(y_test, test_prediction, test_labels)
            test_text = (
                f"\n[test] accuracy: {test_summary['test_accuracy']:.4f}; "
                f"macro-F1: {test_summary['test_macro_f1']:.4f}\n"
                f"Best model fit on train: {model_path}\n"
                + test_report
            )
            print(test_text, flush=True)
            with report_path.open("a", encoding="utf-8") as report_file:
                report_file.write(test_text + "\n")

    results.sort(key=lambda row: row["val_macro_f1"], reverse=True)
    best_config = {**results[0], "best_params": json.loads(results[0]["best_params"])}
    print(f"\nBest on val: {best_config}")

    # Test summaries are printed alongside each pair's train/val results.
    saved_models = "\nBest models fitted on train:\n" + "".join(
        f"- {model_dir / (row['embedding'] + '_' + row['classifier'] + '.joblib')}\n"
        for row in results
    )
    with report_path.open("a", encoding="utf-8") as report_file:
        report_file.write(saved_models)

    test_results_text = "\nTest results by embedding/classifier (best config per pair):\n" + "".join(
        f"- {row['embedding']} + {row['classifier']}: "
        f"accuracy={row['test_accuracy']:.4f}, macro-F1={row['test_macro_f1']:.4f}\n"
        for row in test_summaries
    )
    print(test_results_text, flush=True)
    with report_path.open("a", encoding="utf-8") as report_file:
        report_file.write(test_results_text)

if __name__ == "__main__":
    main()

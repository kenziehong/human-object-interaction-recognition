"""Bước 3: grid search trên val (PredefinedSplit) cho mọi bộ đặc trưng (models.FEATURE_SETS) x 4 mô hình, scoring = macro-F1.

Chạy:  python 02_train_select.py --feat_version f01 --version r01
       python 02_train_select.py --feat_version f01 --version r01 --update --models xgb   (bổ sung vào r01)
Không ghi đè phiên bản đã có trừ khi truyền --update.
"""
import argparse
import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import config as C
import versioning as V
from models import FEATURE_SETS, MODELS, fit_params, load_xy, make_model, rank_results, use_all_cores

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_version", required=True, help="phiên bản đặc trưng trong features/")
    ap.add_argument("--version", required=True, help="phiên bản kết quả -> results/<version>/")
    ap.add_argument("--note", default="", help="mô tả ngắn lần chạy")
    ap.add_argument("--update", action="store_true",
                    help="cho phép bổ sung vào phiên bản đã có (chỉ thay các dòng --feats x --models)")
    ap.add_argument("--feats", nargs="+", default=list(FEATURE_SETS))
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()
    feat_dir = V.feat_dir(args.feat_version)
    if not (feat_dir / "y_train.npy").exists():
        raise SystemExit(f"Chưa có đặc trưng {feat_dir}: chạy 01_extract_features.py --version {args.feat_version}")
    out_dir = V.prepare_dir(V.result_dir(args.version), "val_results.csv", args.update)
    old_info = V.read_run_info(out_dir).get("02_train_select", [])
    if old_info and old_info[0]["args"]["feat_version"] != args.feat_version:
        raise SystemExit(f"{out_dir} dùng đặc trưng {old_info[0]['args']['feat_version']}, không trộn với {args.feat_version}")
    t_start = time.time()
    labels = list(range(len(C.CLASSES)))

    # Chạy một phần (--feats/--models): giữ lại kết quả cũ của các cấu hình khác
    csv = out_dir / "val_results.csv"
    keep = pd.DataFrame()
    if csv.exists():
        old = pd.read_csv(csv)
        keep = old[~(old["feature"].isin(args.feats) & old["model"].isin(args.models))]

    def save(rows):
        df = pd.concat([keep, pd.DataFrame(rows)], ignore_index=True)
        df = rank_results(df)
        df.to_csv(csv, index=False)
        return df

    rows = []
    for feat in args.feats:
        Xtr, ytr = load_xy(feat_dir, "train", feat)
        Xva, yva = load_xy(feat_dir, "val", feat)
        X = np.vstack([Xtr, Xva])
        y = np.concatenate([ytr, yva])
        cv = PredefinedSplit(np.r_[np.full(len(ytr), -1), np.zeros(len(yva))])
        for name in args.models:
            t = time.time()
            pipe, grid = make_model(name, feat, len(ytr), Xtr.shape[1])
            gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=False)
            gs.fit(X, y, **fit_params(name, y))
            grid_s = time.time() - t
            # Fit lại cấu hình tốt nhất CHỈ trên train để lấy chỉ số chi tiết trên val
            best = use_all_cores(pipe.set_params(**gs.best_params_), name)
            t1 = time.perf_counter()
            best.fit(Xtr, ytr, **fit_params(name, ytr))
            fit_s = time.perf_counter() - t1
            pred = best.predict(Xva)
            rec = recall_score(yva, pred, labels=labels, average=None, zero_division=0)
            row = {"feature": feat, "model": name, "dim": Xtr.shape[1],
                   "val_macro_f1": f1_score(yva, pred, labels=labels, average="macro", zero_division=0),
                   "val_balanced_acc": balanced_accuracy_score(yva, pred),
                   **{f"recall_{c}": r for c, r in zip(C.CLASSES, rec)},
                   "best_params": json.dumps(gs.best_params_), "n_grid": len(gs.cv_results_["params"]),
                   "grid_search_s": round(grid_s, 1), "fit_best_s": round(fit_s, 2),
                   "time_s": round(time.time() - t, 1)}
            rows.append(row)
            print(f"{feat:6s} {name:6s} macroF1={row['val_macro_f1']:.4f} "
                  f"balAcc={row['val_balanced_acc']:.4f} {gs.best_params_} ({row['time_s']}s)", flush=True)
            save(rows)

    df = save(rows)
    grids = {m: {k: [str(x) for x in v] for k, v in make_model(m, "-", 1000, 1)[1].items()} for m in args.models}
    V.write_run_info(out_dir, "02_train_select", args, note=args.note, total_wall_s=round(time.time() - t_start, 1),
                     grids=grids,
                     feature_sets={f: FEATURE_SETS[f] for f in args.feats})
    print("\n", df[["feature", "model", "val_macro_f1", "val_balanced_acc"]].to_string(index=False))


if __name__ == "__main__":
    main()

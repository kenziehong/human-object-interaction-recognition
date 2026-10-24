"""Bước 3: grid search trên val (PredefinedSplit) cho mọi bộ đặc trưng (models.FEATURE_SETS) x mô hình, scoring = macro-F1.
clip_zs (zero-shot CLIP) chỉ chạy với bộ đặc trưng "clip"; mặc định chỉ chạy các bộ đặc trưng có đủ khối trong phiên bản.

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
from sklearn.model_selection import GridSearchCV, PredefinedSplit

import resources as R
import versioning as V
from models import (FEATURE_SETS, MODELS, available_feature_sets, eval_metrics, fit_params, load_xy, make_model,
                    model_applies, rank_results, use_all_cores)

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat_version", required=True, help="phiên bản đặc trưng trong data/features/")
    ap.add_argument("--version", required=True, help="phiên bản kết quả -> data/results/<version>/")
    ap.add_argument("--note", default="", help="mô tả ngắn lần chạy")
    ap.add_argument("--update", action="store_true",
                    help="cho phép bổ sung vào phiên bản đã có (chỉ thay các dòng --feats x --models)")
    ap.add_argument("--feats", nargs="+", default=None,
                    help="mặc định: mọi bộ đặc trưng có đủ khối trong phiên bản đặc trưng")
    ap.add_argument("--models", nargs="+", default=MODELS)
    args = ap.parse_args()
    feat_dir = V.feat_dir(args.feat_version)
    if not (feat_dir / "y_train.npy").exists():
        raise SystemExit(f"Chưa có đặc trưng {feat_dir}: chạy 01_extract_features.py --version {args.feat_version}")
    args.feats = args.feats or available_feature_sets(feat_dir)
    out_dir = V.prepare_dir(V.result_dir(args.version), "val_results.csv", args.update)
    old_info = V.read_run_info(out_dir).get("02_train_select", [])
    if old_info and old_info[0]["args"]["feat_version"] != args.feat_version:
        raise SystemExit(f"{out_dir} dùng đặc trưng {old_info[0]['args']['feat_version']}, không trộn với {args.feat_version}")
    t_start = time.time()
    mon = R.ResourceMonitor().start()
    m_all = mon.mark()

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
            if not model_applies(name, feat):
                continue
            t = time.time()
            m = mon.mark()
            pipe, grid = make_model(name, feat, len(ytr), Xtr.shape[1], feat_dir)
            gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=cv, n_jobs=-1, refit=False)
            gs.fit(X, y, **fit_params(name, y))
            grid_s = time.time() - t
            # Điểm val (macro-F1) của từng giá trị trong lưới, ví dụ so riêng T = 0.7 và 0.8 của clip_zs
            cvr = pd.DataFrame(gs.cv_results_)
            cvr[[c for c in cvr if c.startswith("param_")] + ["mean_test_score", "mean_fit_time"]].rename(
                columns={"mean_test_score": "val_macro_f1"}).to_csv(out_dir / f"grid_{feat}_{name}.csv", index=False)
            # Fit lại cấu hình tốt nhất CHỈ trên train để lấy chỉ số chi tiết trên val
            best = use_all_cores(pipe.set_params(**gs.best_params_), name)
            t1 = time.perf_counter()
            best.fit(Xtr, ytr, **fit_params(name, ytr))
            fit_s = time.perf_counter() - t1
            pred = best.predict(Xva)
            np.save(out_dir / f"val_pred_{feat}_{name}.npy", pred)     # để vẽ confusion matrix trên val
            res = mon.since(m)
            row = {"feature": feat, "model": name, "dim": Xtr.shape[1], **eval_metrics(yva, pred, "val"),
                   "best_params": json.dumps(gs.best_params_), "n_grid": len(gs.cv_results_["params"]),
                   "grid_search_s": round(grid_s, 1), "fit_best_s": round(fit_s, 2),
                   "time_s": round(time.time() - t, 1),
                   # RAM đỉnh cả cây tiến trình (gồm worker GridSearchCV) khi grid search + fit lại; X_mb = train+val
                   "peak_rss_mb": res["peak_rss_mb"], "peak_delta_mb": res["peak_delta_mb"], "cpu_s": res["cpu_s"], "cpu_util": res["cpu_util"],
                   "X_mb": round(X.nbytes / R.MB, 1)}
            rows.append(row)
            print(f"{feat:6s} {name:6s} macroF1={row['val_macro_f1']:.4f} "
                  f"balAcc={row['val_balanced_acc']:.4f} acc={row['val_accuracy']:.4f} {gs.best_params_} "
                  f"({row['time_s']}s, RAM đỉnh {row['peak_rss_mb']} MB)", flush=True)
            save(rows)

    df = save(rows)
    res = mon.since(m_all)
    mon.stop()
    grids = {m: {k: [str(x) for x in v] for k, v in make_model(m, "-", 1000, 1)[1].items()} for m in args.models}
    V.write_run_info(out_dir, "02_train_select", args, note=args.note, total_wall_s=round(time.time() - t_start, 1),
                     peak_rss_mb=res["peak_rss_mb"], peak_delta_mb=res["peak_delta_mb"], cpu_s=res["cpu_s"], cpu_util=res["cpu_util"], grids=grids,
                     feature_sets={f: FEATURE_SETS[f] for f in args.feats})
    print("\n", df[["feature", "model", "val_macro_f1", "val_balanced_acc"]].to_string(index=False))


if __name__ == "__main__":
    main()

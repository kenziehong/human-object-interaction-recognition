"""Bước 4: train lại trên train+val với siêu tham số đã chọn trên val, đánh giá test MỘT lần.
Mặc định test MỌI cấu hình trong val_results.csv (từ r03); --best_only chỉ test cấu hình tốt nhất cho từng bộ
đặc trưng và từng mô hình (cách làm của r01, r02). Cấu hình tốt nhất (is_best_on_val) vẫn chọn theo val, riêng
cho từng nhóm phương pháp (models.feature_group): số test chỉ để báo cáo, không dùng để chọn.

Chạy:  python 03_final_test.py --version r01   (đọc data/results/r01/val_results.csv, ghi kết quả test vào data/results/r01/)
Không ghi đè kết quả test đã có trừ khi truyền --update.
"""
import argparse
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

import config as C
import resources as R
import versioning as V
from models import eval_metrics, feature_group, fit_params, load_xy, make_model, rank_results, use_all_cores

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="phiên bản kết quả đã chạy 02_train_select.py")
    ap.add_argument("--note", default="")
    ap.add_argument("--update", action="store_true", help="cho phép ghi đè kết quả test của phiên bản này")
    ap.add_argument("--best_only", action="store_true",
                    help="chỉ test cấu hình tốt nhất cho từng bộ đặc trưng và từng mô hình (mặc định: test mọi cấu hình)")
    args = ap.parse_args()
    out_dir = V.result_dir(args.version)
    info = V.read_run_info(out_dir).get("02_train_select")
    if not info:
        raise SystemExit(f"{out_dir} chưa có kết quả 02_train_select.py")
    feat_dir = V.feat_dir(info[0]["args"]["feat_version"])
    V.prepare_dir(out_dir, "test_results.csv", args.update)
    t_start = time.time()
    mon = R.ResourceMonitor().start()
    m_all = mon.mark()
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(C.CLASSES)))

    val = rank_results(pd.read_csv(out_dir / "val_results.csv"))
    val["group"] = val["feature"].map(feature_group)
    if args.best_only:
        chosen = pd.concat([val.groupby("feature", sort=False).head(1),   # tốt nhất cho mỗi bộ đặc trưng
                            val.groupby("model", sort=False).head(1)])     # tốt nhất cho mỗi mô hình
        chosen = chosen.drop_duplicates(["feature", "model"])
    else:
        chosen = val
    # Tốt nhất trên val trong từng nhóm (val đã xếp hạng nên dòng đầu mỗi nhóm là tốt nhất)
    best_keys = {(r["feature"], r["model"]) for _, r in val.groupby("group", sort=False).head(1).iterrows()}

    rows, report_txt = [], []
    for _, r in chosen.iterrows():
        feat, name = r["feature"], r["model"]
        params = json.loads(r["best_params"])
        Xtr, ytr = load_xy(feat_dir, "train", feat)
        Xva, yva = load_xy(feat_dir, "val", feat)
        Xte, yte = load_xy(feat_dir, "test", feat)
        pipe, _ = make_model(name, feat, len(ytr) + len(yva), Xtr.shape[1], feat_dir)
        use_all_cores(pipe.set_params(**params), name)
        y_all = np.concatenate([ytr, yva])
        X_all = np.vstack([Xtr, Xva])
        m = mon.mark()
        t = time.perf_counter()
        pipe.fit(X_all, y_all, **fit_params(name, y_all))
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        pred = pipe.predict(Xte)
        pred_s = time.perf_counter() - t
        res = mon.since(m)
        joblib.dump(pipe, model_dir / f"{feat}_{name}.joblib")
        np.save(out_dir / f"test_pred_{feat}_{name}.npy", pred)

        rows.append({"feature": feat, "model": name, "group": r["group"], "val_macro_f1": r["val_macro_f1"],
                     **eval_metrics(yte, pred, "test"), "is_best_on_val": (feat, name) in best_keys,
                     "dim": Xtr.shape[1], "fit_s": round(fit_s, 2), "predict_s": round(pred_s, 3),
                     "predict_ms_per_img": round(1000 * pred_s / len(yte), 3),
                     # RAM đỉnh khi fit trên train+val và dự đoán test; model_mb = kích thước file .joblib
                     "peak_rss_mb": res["peak_rss_mb"], "peak_delta_mb": res["peak_delta_mb"], "cpu_util": res["cpu_util"],
                     "X_mb": round(X_all.nbytes / R.MB, 1), "model_mb": R.file_mb(model_dir / f"{feat}_{name}.joblib")})
        rep = classification_report(yte, pred, labels=labels, target_names=C.CLASSES, digits=4, zero_division=0)
        report_txt.append(f"===== {feat} + {name}  params={params} =====\n{rep}")
        print(report_txt[-1], flush=True)

        # Số đếm (hàng = nhãn thật, cột = dự đoán): đọc được số ảnh bị nhầm; lưu cả csv và ảnh PNG
        counts = confusion_matrix(yte, pred, labels=labels)
        pd.DataFrame(counts, index=pd.Index(C.CLASSES, name="true\\pred"), columns=C.CLASSES).to_csv(
            out_dir / f"confusion_{feat}_{name}.csv")
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(counts, display_labels=C.CLASSES).plot(ax=ax, cmap="Blues", values_format="d",
                                                                      xticks_rotation=45, colorbar=False)
        ax.set_title(f"Test confusion (số ảnh): {feat} + {name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_{feat}_{name}.png", dpi=150)
        plt.close(fig)

    pd.DataFrame(rows).to_csv(out_dir / "test_results.csv", index=False)
    (out_dir / "test_report.txt").write_text("\n".join(report_txt), encoding="utf-8")
    res = mon.since(m_all)
    mon.stop()
    V.write_run_info(out_dir, "03_final_test", args, note=args.note, total_wall_s=round(time.time() - t_start, 1),
                     peak_rss_mb=res["peak_rss_mb"], peak_delta_mb=res["peak_delta_mb"], cpu_s=res["cpu_s"], cpu_util=res["cpu_util"],
                     disk_mb=R.dir_mb(out_dir), models_mb=R.dir_mb(model_dir),
                     chosen=[f"{r['feature']}+{r['model']}" for r in rows], best_only=args.best_only,
                     best_on_val=sorted(f"{f}+{m}" for f, m in best_keys))
    print(pd.DataFrame(rows)[["feature", "model", "val_macro_f1", "test_macro_f1", "test_balanced_acc",
                              "test_accuracy", "test_macro_precision", "is_best_on_val", "fit_s", "predict_ms_per_img",
                              "peak_rss_mb", "peak_delta_mb", "model_mb"]].to_string(index=False))


if __name__ == "__main__":
    main()

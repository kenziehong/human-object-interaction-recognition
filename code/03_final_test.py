"""Bước 4: lấy cấu hình tốt nhất theo val (tổng thể, tốt nhất cho từng bộ đặc trưng, tốt nhất cho từng mô hình),
train lại trên train+val, đánh giá test MỘT lần.

Chạy:  python 03_final_test.py --version r01   (đọc results/r01/val_results.csv, ghi kết quả test vào results/r01/)
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
from sklearn.metrics import (ConfusionMatrixDisplay, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score)

import config as C
import versioning as V
from models import fit_params, load_xy, make_model, rank_results, use_all_cores

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="phiên bản kết quả đã chạy 02_train_select.py")
    ap.add_argument("--note", default="")
    ap.add_argument("--update", action="store_true", help="cho phép ghi đè kết quả test của phiên bản này")
    args = ap.parse_args()
    out_dir = V.result_dir(args.version)
    info = V.read_run_info(out_dir).get("02_train_select")
    if not info:
        raise SystemExit(f"{out_dir} chưa có kết quả 02_train_select.py")
    feat_dir = V.feat_dir(info[0]["args"]["feat_version"])
    V.prepare_dir(out_dir, "test_results.csv", args.update)
    t_start = time.time()
    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    labels = list(range(len(C.CLASSES)))

    val = rank_results(pd.read_csv(out_dir / "val_results.csv"))
    chosen = pd.concat([val.groupby("feature", sort=False).head(1),   # tốt nhất cho mỗi bộ đặc trưng
                        val.groupby("model", sort=False).head(1)])     # tốt nhất cho mỗi mô hình
    chosen = chosen.drop_duplicates(["feature", "model"])
    best_key = (val.iloc[0]["feature"], val.iloc[0]["model"])     # tốt nhất tổng thể

    rows, report_txt = [], []
    for _, r in chosen.iterrows():
        feat, name = r["feature"], r["model"]
        params = json.loads(r["best_params"])
        Xtr, ytr = load_xy(feat_dir, "train", feat)
        Xva, yva = load_xy(feat_dir, "val", feat)
        Xte, yte = load_xy(feat_dir, "test", feat)
        pipe, _ = make_model(name, feat, len(ytr) + len(yva), Xtr.shape[1])
        use_all_cores(pipe.set_params(**params), name)
        y_all = np.concatenate([ytr, yva])
        t = time.perf_counter()
        pipe.fit(np.vstack([Xtr, Xva]), y_all, **fit_params(name, y_all))
        fit_s = time.perf_counter() - t
        t = time.perf_counter()
        pred = pipe.predict(Xte)
        pred_s = time.perf_counter() - t
        joblib.dump(pipe, model_dir / f"{feat}_{name}.joblib")
        np.save(out_dir / f"test_pred_{feat}_{name}.npy", pred)

        rec = recall_score(yte, pred, labels=labels, average=None, zero_division=0)
        rows.append({"feature": feat, "model": name, "val_macro_f1": r["val_macro_f1"],
                     "test_macro_f1": f1_score(yte, pred, labels=labels, average="macro", zero_division=0),
                     "test_balanced_acc": balanced_accuracy_score(yte, pred),
                     **{f"recall_{c}": v for c, v in zip(C.CLASSES, rec)}, "is_best_on_val": (feat, name) == best_key,
                     "dim": Xtr.shape[1], "fit_s": round(fit_s, 2), "predict_s": round(pred_s, 3),
                     "predict_ms_per_img": round(1000 * pred_s / len(yte), 3)})
        rep = classification_report(yte, pred, labels=labels, target_names=C.CLASSES, digits=4, zero_division=0)
        report_txt.append(f"===== {feat} + {name}  params={params} =====\n{rep}")
        print(report_txt[-1], flush=True)

        cm = confusion_matrix(yte, pred, labels=labels, normalize="true")
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay(cm, display_labels=C.CLASSES).plot(ax=ax, cmap="Blues", values_format=".2f",
                                                                  xticks_rotation=45, colorbar=False)
        ax.set_title(f"Test confusion (chuẩn hóa theo hàng): {feat} + {name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"confusion_{feat}_{name}.png", dpi=150)
        plt.close(fig)

    pd.DataFrame(rows).to_csv(out_dir / "test_results.csv", index=False)
    (out_dir / "test_report.txt").write_text("\n".join(report_txt), encoding="utf-8")
    V.write_run_info(out_dir, "03_final_test", args, note=args.note, total_wall_s=round(time.time() - t_start, 1),
                     chosen=[f"{r['feature']}+{r['model']}" for r in rows], best_on_val=f"{best_key[0]}+{best_key[1]}")
    print(pd.DataFrame(rows)[["feature", "model", "val_macro_f1", "test_macro_f1", "test_balanced_acc",
                              "is_best_on_val", "fit_s", "predict_ms_per_img"]].to_string(index=False))


if __name__ == "__main__":
    main()

"""Dự đoán hành động KÈM XÁC SUẤT cho ảnh mới bằng CLIP ViT-B/32 + SVM, mặc định mô hình r06_train.

Chỉ NẠP mô hình đã lưu, không fit lại, không đọc data/features/:
  data/models/clip_svm_proba_<phiên bản>.joblib = Pipeline(StandardScaler + SVC RBF, C = 10, γ = scale,
  class_weight=balanced, probability=True), cùng cấu hình với data/results/<phiên bản>/models/clip_svm.joblib.
  - clip_svm_proba_r06_train.joblib: fit trên train (9,069 ảnh), như r06_train.
  - clip_svm_proba_r05.joblib: fit trên train+val (10,947 ảnh), như r05.
  Đã kiểm tra (03/11/2026): nhãn trùng mô hình gốc trên 1,886/1,886 ảnh test (macro-F1 88.8 / 88.9).

Ảnh mới -> đệm vuông -> embedding CLIP 512 chiều (clip_feat.image_embeddings, giống hệt lúc trích đặc trưng)
-> nhãn (predict, giống mô hình gốc) + xác suất 6 lớp (Platt scaling).

Ảnh đầu vào nên là ảnh CẮT quanh một người (như data/dataset_6class_crops).
Lưu ý: CLIP train trên dữ liệu web không công bố, có thể chứa ảnh COCO; kết quả CLIP chỉ là mốc so sánh.

Chạy (trong thư mục code/):
  ../.venv/bin/python predict_clip_svm_prob.py anh.jpg
  ../.venv/bin/python predict_clip_svm_prob.py thu_muc_anh/ --out ket_qua.csv
  ../.venv/bin/python predict_clip_svm_prob.py anh.jpg --version r05
"""
import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import config as C

ROOT = Path(__file__).resolve().parent.parent
MODELS = Path(__file__).resolve().parent / "weights"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(inputs):
    paths = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
        elif p.is_file():
            paths.append(p)
        else:
            sys.exit(f"Không tìm thấy: {s}")
    if not paths:
        sys.exit("Không có ảnh nào để dự đoán.")
    return paths


def main():
    ap = argparse.ArgumentParser(description="Dự đoán hành động kèm xác suất bằng CLIP + SVM")
    ap.add_argument("inputs", nargs="+", help="file ảnh hoặc thư mục ảnh (ảnh crop quanh người)")
    ap.add_argument("--version", default="r06_train", help="phiên bản mô hình (mặc định r06_train; hoặc r05)")
    ap.add_argument("--out", help="lưu kết quả ra file csv")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    model_path = MODELS / f"clip_svm_proba_{args.version}.joblib"
    if not model_path.exists():
        have = ", ".join(p.stem.replace("clip_svm_proba_", "") for p in sorted(MODELS.glob("clip_svm_proba_*.joblib")))
        sys.exit(f"Không có {model_path.relative_to(ROOT)}. Các phiên bản có sẵn: {have or '(không có)'}")
    model = joblib.load(model_path)
    if not getattr(model.named_steps["clf"], "probability", False):
        sys.exit(f"{model_path.name} không bật probability")

    paths = list_images(args.inputs)
    import clip_feat  # import muộn: nạp torch / open_clip chậm

    t0 = time.time()
    X = clip_feat.image_embeddings([str(p) for p in paths], batch_size=args.batch_size)
    t_emb = time.time() - t0

    pred = model.predict(X)              # nhãn lấy từ predict(), giống mô hình gốc
    proba = model.predict_proba(X)
    order = list(model.classes_)
    classes = [C.CLASSES[k] for k in order]
    rows = []
    for p, k, pr in zip(paths, pred, proba):
        top = np.argsort(pr)[::-1][:3]
        r = {"image": str(p), "pred": C.CLASSES[k], "confidence": round(float(pr[order.index(k)]), 4),
             "top3": ", ".join(f"{classes[j]} {100 * pr[j]:.1f}%" for j in top)}
        r.update({f"p_{c}": round(float(v), 4) for c, v in zip(classes, pr)})
        rows.append(r)

    w = max(len(Path(r["image"]).name) for r in rows)
    print(f"Mô hình: {model_path.relative_to(ROOT)}")
    for r in rows:
        print(f"{Path(r['image']).name:{w}s}  ->  {r['pred']} ({100 * r['confidence']:.1f}%)   top-3: {r['top3']}")
    print(f"\n{len(paths)} ảnh, embedding CLIP {t_emb:.1f} s, tổng {time.time() - t0:.1f} s")
    df = pd.DataFrame(rows)
    if args.out:
        df.to_csv(args.out, index=False)
        print("đã lưu", args.out)
    return df


if __name__ == "__main__":
    main()

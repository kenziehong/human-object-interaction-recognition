"""Dự đoán hành động cho ảnh mới bằng CLIP ViT-B/32 + SVM (mô hình đã chọn trên val ở r05).

Đầu vào: một hoặc nhiều ảnh, hoặc thư mục ảnh. Ảnh nên là ảnh CẮT quanh người (như data/dataset_6class_crops),
vì mô hình được train trên ảnh crop người; ảnh toàn cảnh nhiều người sẽ cho kết quả kém tin cậy.

Cách làm (giống hệt lúc train):
  ảnh -> đệm thành hình vuông -> CLIP ViT-B/32 (open_clip, 'openai') -> embedding 512 chiều chuẩn hoá L2
      -> Pipeline(StandardScaler + SVC RBF) đã lưu ở data/results/r05/models/clip_svm.joblib -> nhãn.

Độ tin cậy (--proba): mô hình r05 không bật probability nên không có predict_proba. Với --proba, script dùng
bản fit lại đúng cấu hình đó với probability=True (Platt scaling) trên train+val của f03, lưu ở
data/models/clip_svm_proba_r05.joblib. Lần đầu chưa có file thì tự fit (~40 s) rồi lưu; các lần sau chỉ nạp.
Đã kiểm tra (03/11/2026): bản này cho nhãn trùng r05 trên 1,886/1,886 ảnh test, test macro-F1 88.9.
Nhãn in ra luôn lấy từ predict() (giống r05); xác suất chỉ để tham khảo độ tin cậy.

Lưu ý: CLIP train trên dữ liệu web không công bố, có thể chứa ảnh COCO; kết quả CLIP chỉ là mốc so sánh.

Chạy (trong thư mục code/):
  ../.venv/bin/python predict_clip_svm.py anh1.jpg anh2.jpg
  ../.venv/bin/python predict_clip_svm.py thu_muc_anh/ --proba --out ket_qua.csv
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
MODEL_R05 = ROOT / "data" / "results" / "r05" / "models" / "clip_svm.joblib"
FEAT_F03 = ROOT / "data" / "features" / "f03"
MODEL_PROBA = ROOT / "data" / "models" / "clip_svm_proba_r05.joblib"
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


def load_proba_model():
    """Nạp (hoặc fit lần đầu rồi lưu) bản SVM có predict_proba, cùng cấu hình với r05."""
    if MODEL_PROBA.exists():
        return joblib.load(MODEL_PROBA)
    from sklearn.base import clone
    print(f"Chưa có {MODEL_PROBA.relative_to(ROOT)}: fit SVM probability=True trên train+val f03 (~40 s)...")
    t0 = time.time()
    model = clone(joblib.load(MODEL_R05))
    model.set_params(clf__probability=True, clf__random_state=C.SEED)
    X = np.vstack([np.load(FEAT_F03 / f"{s}_clip.npy").astype(np.float32) for s in ("train", "val")])
    y = np.concatenate([np.load(FEAT_F03 / f"y_{s}.npy") for s in ("train", "val")])
    model.fit(X, y)
    MODEL_PROBA.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PROBA)
    print(f"  xong sau {time.time() - t0:.1f} s, đã lưu {MODEL_PROBA.relative_to(ROOT)}")
    return model


def main():
    ap = argparse.ArgumentParser(description="Dự đoán hành động bằng CLIP + SVM (r05)")
    ap.add_argument("inputs", nargs="+", help="file ảnh hoặc thư mục ảnh (ảnh crop quanh người)")
    ap.add_argument("--proba", action="store_true", help="in thêm độ tin cậy và top-3 lớp")
    ap.add_argument("--out", help="lưu kết quả ra file csv")
    ap.add_argument("--batch_size", type=int, default=64)
    args = ap.parse_args()

    paths = list_images(args.inputs)
    import clip_feat  # import muộn: nạp torch / open_clip chậm

    t0 = time.time()
    X = clip_feat.image_embeddings([str(p) for p in paths], batch_size=args.batch_size)
    t_emb = time.time() - t0

    model = load_proba_model() if args.proba else joblib.load(MODEL_R05)
    pred = model.predict(X)
    rows = []
    for i, p in enumerate(paths):
        r = {"image": str(p), "pred": C.CLASSES[pred[i]]}
        rows.append(r)
    if args.proba:
        proba = model.predict_proba(X)
        classes = [C.CLASSES[k] for k in model.classes_]
        for r, pr, k in zip(rows, proba, pred):
            r["confidence"] = round(float(pr[list(model.classes_).index(k)]), 4)
            top = np.argsort(pr)[::-1][:3]
            r["top3"] = ", ".join(f"{classes[j]} {100 * pr[j]:.1f}%" for j in top)
            for c, v in zip(classes, pr):
                r[f"p_{c}"] = round(float(v), 4)

    df = pd.DataFrame(rows)
    w = max(len(Path(r["image"]).name) for r in rows)
    for r in rows:
        line = f"{Path(r['image']).name:{w}s}  ->  {r['pred']}"
        if args.proba:
            line += f" ({100 * r['confidence']:.1f}%)   top-3: {r['top3']}"
        print(line)
    print(f"\n{len(paths)} ảnh, embedding CLIP {t_emb:.1f} s, tổng {time.time() - t0:.1f} s")
    if args.out:
        df.to_csv(args.out, index=False)
        print("đã lưu", args.out)
    return df


if __name__ == "__main__":
    main()

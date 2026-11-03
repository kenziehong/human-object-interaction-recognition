"""Vẽ Hình 8 của báo cáo: dự đoán CLIP + SVM kèm top-3 xác suất trên 12 ảnh test của Hình 7.

Mô hình CLIP + SVM ở r05 (data/results/r05/models/clip_svm.joblib) không bật probability nên
không có predict_proba. Script này fit lại đúng cấu hình đã chọn trên val (StandardScaler +
SVC RBF, siêu tham số đọc từ joblib r05) với probability=True (Platt scaling, CV 5-fold nội bộ),
trên train+val như bước test cuối, rồi:
  - so nhãn dự đoán với r05 trên toàn bộ test (báo số ảnh lệch) và in macro-F1 test;
  - vẽ 12 ảnh, mỗi ảnh ghi Thật / Dự đoán (xác suất) + top-3 lớp, theo kiểu Hình 7.
Không ghi gì vào data/results/ (chỉ đọc). Ảnh xuất ra doc/.

Chạy: cd code && ../.venv/bin/python fig_clip_svm_proba.py
"""
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.metrics import f1_score

import config as C

ROOT = Path(__file__).resolve().parent.parent
FEAT = ROOT / "data" / "features" / "f03"
R05 = ROOT / "data" / "results" / "r05"
CROPS = ROOT / "data" / "dataset_6class_crops"
OUT = ROOT / "doc" / "hinh8_clip_svm_top3.png"

# 12 ảnh test của Hình 7/8 (cột = lớp thật theo C.CLASSES; hàng trên: ResNet50 đúng, hàng dưới: ResNet50 sai),
# xác định bằng so khớp ảnh với meta_test.csv
IMAGES = [
    "hold/vcoco_24457_502941.jpg", "ride/voc_2012_003700_0.jpg", "eat_drink/vcoco_80932_188312.jpg",
    "talk_on_phone/vcoco_279696_506571.jpg", "work_on_computer/vcoco_153563_493356.jpg",
    "unknown/vcoco_192838_427629.jpg",
    "hold/vcoco_404823_204799.jpg", "ride/vcoco_40399_475537.jpg", "eat_drink/vcoco_392055_435749.jpg",
    "talk_on_phone/vcoco_219558_1249643.jpg", "work_on_computer/vcoco_578572_437891.jpg",
    "unknown/vcoco_65655_568981.jpg",
]


def main():
    t0 = time.time()
    X = {s: np.load(FEAT / f"{s}_clip.npy").astype(np.float32) for s in C.SPLITS}
    y = {s: np.load(FEAT / f"y_{s}.npy") for s in C.SPLITS}
    meta = pd.read_csv(FEAT / "meta_test.csv")

    r05 = joblib.load(R05 / "models" / "clip_svm.joblib")
    model = clone(r05)
    model.set_params(clf__probability=True, clf__random_state=C.SEED)
    print("Cấu hình:", model)
    model.fit(np.vstack([X["train"], X["val"]]), np.concatenate([y["train"], y["val"]]))
    print(f"fit: {time.time() - t0:.1f} s")

    proba = model.predict_proba(X["test"])
    pred = model.predict(X["test"])
    pred_r05 = np.load(R05 / "test_pred_clip_svm.npy")
    f1 = f1_score(y["test"], pred, average="macro")
    print(f"test macro-F1 (probability=True): {100 * f1:.1f}  | r05: "
          f"{100 * f1_score(y['test'], pred_r05, average='macro'):.1f}")
    print(f"nhãn lệch so với r05: {(pred != pred_r05).sum()} / {len(pred)} ảnh")

    idx = [int(meta.index[meta.crop_path == p][0]) for p in IMAGES]
    fig, axes = plt.subplots(2, 6, figsize=(19, 9.4))
    for ax, i in zip(axes.ravel(), idx):
        p = proba[i]
        top = np.argsort(p)[::-1][:3]
        true, guess = int(y["test"][i]), int(pred[i])
        ok = true == guess
        color = "#1e7b34" if ok else "#c62828"
        img = Image.open(CROPS / meta.crop_path[i]).convert("RGB")
        cw, ch = 340, 400  # ô dọc như Hình 7; phóng/thu ảnh cho vừa ô, giữ tỉ lệ
        k = min(cw / img.width, ch / img.height)
        img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
        canvas = Image.new("RGB", (cw, ch), "white")
        canvas.paste(img, ((cw - img.width) // 2, (ch - img.height) // 2))
        ax.imshow(canvas)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor(color); s.set_linewidth(3.5)
        ax.set_title(f"Thật: {C.CLASSES[true]}\nDự đoán: {C.CLASSES[guess]} ({100 * p[guess]:.0f}%)",
                     color=color, fontsize=10, fontweight="bold")
        ax.set_xlabel(" · ".join(f"{C.CLASSES[k]} {100 * p[k]:.0f}%" for k in top), fontsize=7.5)
        if guess != top[0]:
            print(f"  lưu ý: {meta.crop_path[i]} predict() khác argmax(proba)")
    fig.suptitle("Kết quả định tính - CLIP ViT-B/32 + SVM trên tập test (viền xanh: đúng, viền đỏ: sai)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor="white")
    print("đã lưu", OUT)
    for i in idx:
        p = proba[i]
        print(f"{meta.crop_path[i]:45s} thật={C.CLASSES[y['test'][i]]:17s} đoán={C.CLASSES[pred[i]]:17s} "
              + ", ".join(f"{C.CLASSES[k]} {100 * p[k]:.0f}%" for k in np.argsort(p)[::-1][:3]))


if __name__ == "__main__":
    main()

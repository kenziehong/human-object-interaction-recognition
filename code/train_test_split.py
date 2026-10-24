"""Chia data/dataset_6class_crops/labels.csv thành train/val/test (mặc định 70/15/15, seed 2026).

- Chia theo ảnh gốc: mọi crop của cùng một ảnh (dataset + source_image) nằm chung 1 split,
  để 2 người trên cùng ảnh không rơi vào train và test.
- Phân tầng theo lớp × nguồn, để mỗi split có tỉ lệ lớp và nguồn gần giống nhau.
- Cách làm: StratifiedGroupKFold 20 fold (mỗi fold 5%), gán 14 fold -> train, 3 -> val, 3 -> test.
- --max_unknown N: sau khi chia, xáo các dòng `unknown` theo seed rồi giữ N dòng đầu (tính trên cả 3 split);
  các lớp khác giữ nguyên.
- Không sửa labels.csv; ghi file mới vào data/train_test_split/: labels_split.csv,
  hoặc labels_split_unk<N>.csv; có thêm cột `split`, không ghi đè file đã có.
- crop_path giữ nguyên (tính từ data/dataset_6class_crops/), nên khi dùng file split phải chỉ thư mục ảnh:
  01_extract_features.py --labels data/train_test_split/labels_split_unk3000.csv --data_dir data/dataset_6class_crops

Chạy (trong code/): ../.venv/bin/python train_test_split.py [--seed 2026] [--ratios 0.7 0.15 0.15] [--max_unknown 3000]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

import config as C

DEFAULT_LABELS = C.ROOT / "data" / "dataset_6class_crops" / "labels.csv"
OUT_DIR = C.ROOT / "data" / "train_test_split"
N_FOLDS = 20


def split_labels(df, seed, ratios):
    n_fold = [round(r * N_FOLDS) for r in ratios]
    assert sum(n_fold) == N_FOLDS, f"Tỉ lệ {ratios} phải là bội của {1 / N_FOLDS:.0%} và cộng lại bằng 1"
    groups = df["dataset"] + "/" + df["source_image"]
    strata = df["label"] + "|" + df["dataset"]
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    fold = np.empty(len(df), dtype=int)
    for k, (_, idx) in enumerate(sgkf.split(df, strata, groups)):
        fold[idx] = k
    # Thứ tự fold cũng được xáo theo seed, để split không phụ thuộc thứ tự fold của sklearn
    order = np.random.default_rng(seed).permutation(N_FOLDS)
    names = np.repeat(C.SPLITS, n_fold)
    fold_to_split = dict(zip(order, names))
    out = df.copy()
    out["split"] = [fold_to_split[f] for f in fold]
    assert out.groupby(groups)["split"].nunique().max() == 1, "Có ảnh gốc nằm ở 2 split"
    return out


def limit_unknown(df, n, seed):
    """Xáo các dòng unknown theo seed, giữ n dòng đầu; thứ tự các dòng còn lại như trong labels.csv."""
    unk = df.index[df["label"] == "unknown"].to_numpy()
    shuffled = np.random.default_rng(seed).permutation(unk)
    drop = shuffled[n:]
    return df.drop(index=drop).reset_index(drop=True)


def summary(df):
    tab = pd.crosstab(df["label"], df["split"], margins=True, margins_name="tổng")
    tab = tab[C.SPLITS + ["tổng"]].reindex(C.CLASSES + ["tổng"])
    pct = (tab[C.SPLITS].div(tab["tổng"], axis=0) * 100).round(1).add_suffix(" %")
    return pd.concat([tab, pct], axis=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    ap.add_argument("--out", type=Path, default=None, help=f"mặc định: {OUT_DIR.relative_to(C.ROOT)}/labels_split*.csv")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--max_unknown", type=int, default=None,
                    help="sau khi chia, xáo unknown theo seed và giữ N dòng đầu (mặc định: giữ hết)")
    args = ap.parse_args()
    name = "labels_split.csv" if args.max_unknown is None else f"labels_split_unk{args.max_unknown}.csv"
    out_path = args.out or OUT_DIR / name
    if out_path.exists():
        raise SystemExit(f"{out_path} đã có, không ghi đè. Đặt --out khác.")

    df = pd.read_csv(args.labels)
    assert df["label"].isin(C.CLASSES).all(), "Có nhãn lạ trong labels.csv"
    out = split_labels(df, args.seed, args.ratios)
    if args.max_unknown is not None:
        out = limit_unknown(out, args.max_unknown, args.seed)
    out.to_csv(out_path, index=False)

    print(f"Đọc {len(df)} crop từ {args.labels}; seed={args.seed}, tỉ lệ={args.ratios}")
    print(f"Số ảnh gốc: {(df['dataset'] + '/' + df['source_image']).nunique()}")
    if args.max_unknown is not None:
        print(f"Giữ {args.max_unknown} unknown đầu tiên sau khi xáo -> còn {len(out)} crop")
    print()
    print("Theo lớp:\n", summary(out).to_string(), "\n")
    print("Theo nguồn:\n", pd.crosstab(out["dataset"], out["split"], margins=True, margins_name="tổng")
          [C.SPLITS + ["tổng"]].to_string(), "\n")
    print(f"Đã ghi {out_path}")


if __name__ == "__main__":
    main()

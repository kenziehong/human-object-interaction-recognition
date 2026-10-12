"""Đọc labels.csv và ảnh crop."""
import os

import cv2
import pandas as pd

from config import CLASS_TO_ID, DATA_DIR, LABELS_CSV


def load_labels(split=None, only_existing=True):
    df = pd.read_csv(LABELS_CSV)
    df["label_id"] = df["label"].map(CLASS_TO_ID)
    assert df["label_id"].notna().all(), "Có nhãn lạ trong labels.csv"
    df["path"] = df["crop_path"].map(lambda p: str(DATA_DIR / p))
    if split is not None:
        df = df[df["split"] == split].reset_index(drop=True)
    if only_existing:
        exists = df["path"].map(os.path.exists)
        if not exists.all():
            print(f"Cảnh báo: [{split or 'all'}] thiếu {(~exists).sum()}/{len(df)} ảnh trên đĩa -> bỏ qua")
            df = df[exists].reset_index(drop=True)
    return df


def read_image(path):
    """Trả về ảnh BGR (uint8)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img

"""Đọc labels.csv và ảnh crop."""
import os
from pathlib import Path

import cv2
import pandas as pd

import config as C


def _abs(path):
    path = Path(path)
    return path if path.is_absolute() else C.ROOT / path


def set_labels_csv(path, data_dir=None):
    """Đổi file nhãn cho cả pipeline. Ảnh được tìm theo crop_path tính từ data_dir
    (mặc định: thư mục chứa file nhãn)."""
    path = _abs(path)
    data_dir = _abs(data_dir) if data_dir else path.parent
    for p in (path, data_dir):
        if not p.exists():
            raise SystemExit(f"Không thấy {p}")
    C.LABELS_CSV, C.DATA_DIR = path, data_dir
    return path


def load_labels(split=None, only_existing=True):
    df = pd.read_csv(C.LABELS_CSV)
    df["label_id"] = df["label"].map(C.CLASS_TO_ID)
    assert df["label_id"].notna().all(), "Có nhãn lạ trong labels.csv"
    df["path"] = df["crop_path"].map(lambda p: str(C.DATA_DIR / p))
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

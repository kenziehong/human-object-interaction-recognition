"""CSV-backed image dataset, integrity checks, and aspect-preserving transforms."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from PIL import Image, ImageOps, UnidentifiedImageError
from torch.utils.data import Dataset
from torchvision import transforms

try:  # Support both `python deep_learning/train.py` and package imports.
    from .config import (CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
                         LABEL_TO_ID, ROOT)
except ImportError:  # pragma: no cover - direct script entry points.
    from config import (CLASS_NAMES, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD,
                        LABEL_TO_ID, ROOT)

CSV_PATH = ROOT / "data/train_test_split/labels_split_unk3000.csv"
IMAGE_ROOT = ROOT / "data/dataset_6class_crops_v2"
SPLITS = ("train", "val", "test")


class PadToSquare:
    """Resize within a square canvas without discarding any image content."""
    def __init__(self, size: int, fill=(0, 0, 0)):
        self.size = size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        scale = self.size / max(width, height)
        resized = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                               Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, ((self.size - resized.width) // 2, (self.size - resized.height) // 2))
        return canvas


def make_transform():
    return transforms.Compose([
        PadToSquare(IMAGE_SIZE), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _image_path(crop_path: str, image_root: Path = IMAGE_ROOT) -> Path:
    parts = Path(str(crop_path)).parts
    if Path(str(crop_path)).is_absolute() or ".." in parts:
        raise ValueError(f"crop_path phải là đường dẫn tương đối an toàn: {crop_path}")
    return image_root.joinpath(*parts)


def load_manifest(csv_path: Path = CSV_PATH, image_root: Path = IMAGE_ROOT,
                  check_images: bool = True) -> pd.DataFrame:
    """Validate the authoritative split CSV and return a manifest with paths/IDs."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy CSV split: {csv_path}")
    frame = pd.read_csv(csv_path)
    required = {"label", "crop_path", "split", "dataset"}
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"CSV thiếu cột: {', '.join(sorted(missing_columns))}")
    frame["label"] = frame["label"].astype(str)
    frame["split"] = frame["split"].astype(str)
    invalid_labels = sorted(set(frame["label"]) - set(CLASS_NAMES))
    invalid_splits = sorted(set(frame["split"]) - set(SPLITS))
    duplicates = int(frame["crop_path"].duplicated(keep=False).sum())
    if invalid_labels or invalid_splits or duplicates:
        raise ValueError(f"CSV không hợp lệ: nhãn={invalid_labels}, split={invalid_splits}, "
                         f"dòng có crop_path trùng={duplicates}.")
    if "source_image" in frame and frame["source_image"].notna().any():
        source_counts = frame.groupby("source_image")["split"].nunique()
        crossed = source_counts[source_counts > 1].index.tolist()
        if crossed:
            raise ValueError(f"{len(crossed)} source_image xuất hiện ở nhiều split; ví dụ: {crossed[:5]}")
    frame["label_id"] = frame["label"].map(LABEL_TO_ID).astype(int)
    frame["image_path"] = frame["crop_path"].map(lambda value: _image_path(value, image_root))
    missing = [str(path) for path in frame["image_path"] if not path.is_file()]
    bad_images = []
    if check_images:
        for path in frame["image_path"]:
            if path.is_file():
                try:
                    with Image.open(path) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError, ValueError):
                    bad_images.append(str(path))
    print(f"CSV: {len(frame)} dòng; ảnh thiếu: {len(missing)}; ảnh lỗi: {len(bad_images)}; "
          f"crop_path trùng: {duplicates}; nhãn/split lỗi: 0")
    if missing or bad_images:
        details = (missing + bad_images)[:10]
        raise FileNotFoundError("Ảnh thiếu/không đọc được; ví dụ: " + ", ".join(details))
    summary = frame.groupby(["split", "label", "dataset"], observed=True).size().rename("count")
    print(summary.to_string())
    for split in SPLITS:
        labels = set(frame.loc[frame.split == split, "label"])
        if labels != set(CLASS_NAMES):
            raise ValueError(f"Split '{split}' không có đủ sáu lớp: thiếu {sorted(set(CLASS_NAMES)-labels)}")
    return frame


class InteractionImageDataset(Dataset):
    """Load prepared RGB images without cropping or modifying their source files."""
    def __init__(self, manifest: pd.DataFrame, split: str, transform=None):
        self.rows = manifest.loc[manifest["split"] == split].reset_index(drop=True)
        if self.rows.empty:
            raise ValueError(f"Split '{split}' không có ảnh.")
        self.transform = transform or make_transform()

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows.iloc[index]
        path = Path(row.image_path)
        try:
            with Image.open(path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
        except (OSError, UnidentifiedImageError) as exc:
            raise RuntimeError(f"Không đọc được ảnh '{path}': {exc}") from exc
        return self.transform(image), int(row.label_id)


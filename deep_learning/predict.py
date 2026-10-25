"""Predict one or more image crops and emit schema-compatible JSON lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageOps

from config import CLASS_NAMES, select_device
from data import make_transform
from model import build_model


def predict_image(model, image_path: Path, transform, device, class_names):
    try:
        with Image.open(image_path) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
    except OSError as exc:
        raise RuntimeError(f"Không đọc được ảnh '{image_path}': {exc}") from exc
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    confidence, label_id = probabilities.max(dim=0)
    index = int(label_id.item())
    return {"image_path": str(image_path), "label_id": index, "label": class_names[index],
            "confidence": float(confidence.item())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Đường dẫn tới best.pt")
    parser.add_argument("--image", type=Path, nargs="+", required=True)
    parser.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(f"Không tìm thấy checkpoint: {args.model}")
    checkpoint = torch.load(args.model, map_location="cpu", weights_only=False)
    class_names = checkpoint.get("class_names", list(CLASS_NAMES))
    if tuple(class_names) != CLASS_NAMES:
        raise ValueError("Class order trong checkpoint không khớp với pipeline.")
    model = build_model(checkpoint["backbone"], pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    device = select_device(args.device)
    model.to(device).eval()
    transform = make_transform()
    for image_path in args.image:
        result = predict_image(model, image_path, transform, device, class_names)
        result["model_name"] = checkpoint["model_name"]
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

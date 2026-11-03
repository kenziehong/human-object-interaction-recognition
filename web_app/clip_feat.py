"""Khối đặc trưng CLIP: mốc so sánh, KHÔNG phải đặc trưng thủ công (nhóm 'clip', xem models.feature_group).

- Embedding ảnh: CLIP ViT-B/32 (open_clip, trọng số 'openai'), chuẩn hoá L2, 512 chiều.
- Embedding văn bản: 1 prompt cho mỗi lớp hành động (config.CLIP_PROMPTS), dùng cho zero-shot (models.ClipZeroShot).
- Ảnh crop người được ĐỆM thành hình vuông trước khi đưa vào CLIP: tiền xử lý mặc định cắt giữa về 224x224 sẽ làm
  mất đầu/chân của crop dáng đứng (đúng chỗ có điện thoại, đồ ăn, ngựa).
- Lưu ý dữ liệu: CLIP train trên ~400 triệu cặp ảnh-văn bản lấy từ web (không công bố), không loại trừ được ảnh COCO.

torch / open_clip chỉ được import khi gọi các hàm ở đây (01_extract_features.py --clip), nên pipeline đặc trưng
thủ công không cần cài torch.
"""
import cv2
import numpy as np

import config as C
from data_utils import read_image

_model = None


def _load():
    global _model
    if _model is None:
        import open_clip
        import torch
        torch.set_num_threads(C.CLIP_THREADS)
        model, _, preprocess = open_clip.create_model_and_transforms(C.CLIP_MODEL, pretrained=C.CLIP_PRETRAINED)
        model.eval()
        _model = (model, preprocess, open_clip.get_tokenizer(C.CLIP_MODEL), torch)
    return _model


def pad_square(rgb):
    """Đệm ảnh RGB thành hình vuông bằng màu trung bình của CLIP, giữ nguyên toàn bộ người."""
    h, w = rgb.shape[:2]
    s = max(h, w)
    out = np.empty((s, s, 3), np.uint8)
    out[:] = C.CLIP_PAD_RGB
    y, x = (s - h) // 2, (s - w) // 2
    out[y:y + h, x:x + w] = rgb
    return out


def image_embeddings(paths, batch_size=64):
    """(n, 512) float32, đã chuẩn hoá L2. Chạy trên CPU để kết quả tái lập được."""
    from PIL import Image
    model, preprocess, _, torch = _load()
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch_size):
            ims = [preprocess(Image.fromarray(pad_square(cv2.cvtColor(read_image(p), cv2.COLOR_BGR2RGB))))
                   for p in paths[i:i + batch_size]]
            e = model.encode_image(torch.stack(ims))
            out.append((e / e.norm(dim=-1, keepdim=True)).numpy())
    return np.concatenate(out).astype(np.float32)


def text_embeddings():
    """(số lớp hành động, 512) float32, đã chuẩn hoá L2; hàng i ứng với C.CLASSES[i]."""
    model, _, tokenizer, torch = _load()
    with torch.no_grad():
        e = model.encode_text(tokenizer([C.CLIP_PROMPTS[c] for c in C.CLASSES if c in C.CLIP_PROMPTS]))
    return (e / e.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)

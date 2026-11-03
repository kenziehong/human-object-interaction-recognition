"""Phân loại crop người: embedding CLIP ViT-B/32 (open_clip, 'openai') + StandardScaler + SVC (weights/clip_svm_proba_r06_train.joblib).

Độ tin cậy (giống predict_clip_svm_prob.py): mô hình clip_svm_proba_r06_train.joblib fit với probability=True
(train 9,069 ảnh, nhãn trùng r06_train trên 1,886/1,886 ảnh test) nên nhãn lấy từ predict(), xác suất từ
predict_proba().

Tiền xử lý giống hệt lúc train (code/clip_feat.py trên nhánh main): đệm crop thành hình vuông bằng màu trung bình
của CLIP, rồi preprocess mặc định của open_clip, embedding chuẩn hoá L2 (512 chiều). Chạy CLIP trên CPU.
"""
import os
import threading

import joblib
import numpy as np
import open_clip
import torch
from PIL import Image

SVM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'clip_svm_proba_r06_train.joblib')
CLASSES = ['hold', 'ride', 'eat/drink', 'talk_on_phone', 'work_on_computer', 'unknown']   # = config.CLASSES
CLIP_MODEL, CLIP_PRETRAINED = 'ViT-B-32', 'openai'
CLIP_PAD_RGB = (122, 116, 104)   # màu đệm = trung bình chuẩn hoá của CLIP × 255
TEN_LOP_VN = {'hold': 'Cầm / giữ vật', 'ride': 'Cưỡi / đi xe', 'eat/drink': 'Ăn / uống',
              'talk_on_phone': 'Nghe điện thoại', 'work_on_computer': 'Dùng máy tính', 'unknown': 'Không xác định'}


def pad_square(rgb):
    """Đệm ảnh RGB (H, W, 3) thành hình vuông, giữ nguyên toàn bộ người (center-crop sẽ cắt mất đầu/chân)."""
    h, w = rgb.shape[:2]
    s = max(h, w)
    out = np.empty((s, s, 3), np.uint8)
    out[:] = CLIP_PAD_RGB
    y, x = (s - h) // 2, (s - w) // 2
    out[y:y + h, x:x + w] = rgb
    return out


class PhanLoai:
    """danh sách crop PIL → [{'lop', 'ten_vn', 'do_tin_cay', 'top3', 'xac_suat': [{'lop', 'ten_vn', 'p'} từng lớp]}].

    do_tin_cay / top3 / xac_suat = None khi file .joblib không bật probability.
    """

    def __init__(self, svm_path=SVM_PATH, threads=8):
        torch.set_num_threads(threads)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(CLIP_MODEL, pretrained=CLIP_PRETRAINED)
        self.model.eval()
        self.svm = joblib.load(svm_path)
        self.co_proba = bool(self.svm.named_steps['clf'].probability)
        print('Phân loại:', os.path.basename(svm_path),
              '(có xác suất)' if self.co_proba else '(probability=False: chỉ có nhãn, không có xác suất)')
        self.lock = threading.Lock()

    def __call__(self, crops):
        x = torch.stack([self.preprocess(Image.fromarray(pad_square(np.asarray(c.convert('RGB'))))) for c in crops])
        with self.lock, torch.no_grad():
            e = self.model.encode_image(x)
        X = (e / e.norm(dim=-1, keepdim=True)).numpy().astype(np.float32)
        pred = self.svm.predict(X)
        out = [{'lop': CLASSES[k], 'ten_vn': TEN_LOP_VN[CLASSES[k]], 'do_tin_cay': None, 'top3': None, 'xac_suat': None}
               for k in pred]
        if self.co_proba:
            ks = list(self.svm.classes_)
            proba = self.svm.predict_proba(X)
            for r, pr, k in zip(out, proba, pred):
                r['do_tin_cay'] = round(float(pr[ks.index(k)]), 4)
                xs = [{'lop': CLASSES[c], 'ten_vn': TEN_LOP_VN[CLASSES[c]], 'p': round(float(v), 4)} for c, v in zip(ks, pr)]
                r['xac_suat'] = xs
                r['top3'] = sorted(xs, key=lambda d: -d['p'])[:3]
        return out

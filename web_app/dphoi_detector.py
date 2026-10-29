"""Tiền xử lý lúc inference: dùng mô hình DP-HOI (CVPR 2024) tìm người trong ảnh gốc, rồi cắt crop người
giống hệt cách tạo dataset_6class_crops (bbox người nới 20% mỗi phía, không vượt ra ngoài ảnh).

Mô hình phân loại 6 lớp được train trên ảnh "đúng một người chính" (xi = ảnh cắt theo bbox người). Ảnh thực tế
lại là ảnh nguyên, nên phải có bước này đứng trước:  ảnh gốc → DP-HOI tìm người → cắt → mô hình phân loại.

Trọng số: checkpoint pre-trained DP-HOI ResNet50 (DETR, 3 lớp decoder, 100 query, 91 lớp COCO):
    weights/dphoi_res50_3layers.pth      (tải bằng scripts/setup.sh)
Chỉ dùng nhánh phát hiện (detection) của checkpoint; nhánh action/caption chỉ dùng lúc pre-train nên bỏ qua,
vì vậy không cần cài CLIP và chạy được trên CPU / Apple MPS (code gốc gọi .cuda() cứng nên không dùng thẳng được).

Cần: pip install -r requirements.txt

    python dphoi_detector.py anh.jpg                      # cắt mọi người, lưu vào crops_dphoi/
    python dphoi_detector.py thu_muc_anh/ --out ket_qua/ --ve
    python dphoi_detector.py anh.jpg --mot-nguoi          # chỉ lấy 1 người chính

Dùng trong code (trước mô hình phân loại):
    from dphoi_detector import CatNguoi
    cat = CatNguoi()
    for crop, box, score in cat(Image.open('anh.jpg')):
        nhan = mo_hinh_phan_loai(crop)
"""
import argparse, math, os, sys, types

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.abspath(__file__))
DPHOI = os.path.join(ROOT, 'third_party', 'DP-HOI')          # git clone, xem README
CKPT = os.path.join(ROOT, 'weights', 'dphoi_res50_3layers.pth')

PAD = 0.20           # nới bbox người 20% mỗi phía, giống tao_dataset_6class.py
PERSON = 1           # id 'person' trong 91 lớp COCO
NUM_CLASSES = 91
NUM_QUERIES = 100
MIN_SIZE, MAX_SIZE = 800, 1333   # resize giống lúc đánh giá của DETR/DP-HOI (T.RandomResize([800], max_size=1333))
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
DUOI_ANH = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

# models/__init__ kéo theo clip_model.py (import clip) dù inference không dùng tới → thay bằng module rỗng
sys.modules.setdefault('clip', types.ModuleType('clip'))
sys.path.insert(0, DPHOI)
from models.backbone import BackboneBase, FrozenBatchNorm2d, Joiner          # noqa: E402
from models.position_encoding import build_position_encoding                 # noqa: E402
from models.transformer import build_transformer                             # noqa: E402
from models.detr import DETR                                                 # noqa: E402
from util.misc import nested_tensor_from_tensor_list                         # noqa: E402
from util.box_ops import box_cxcywh_to_xyxy                                  # noqa: E402


def khung_cat(bbox, W, H):
    """bbox người (x1,y1,x2,y2) → khung cắt đã nới PAD mỗi phía, không vượt ra ngoài ảnh (copy từ tao_dataset_6class.py)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (max(0, math.floor(x1 - PAD * w)), max(0, math.floor(y1 - PAD * h)),
            min(W, math.ceil(x2 + PAD * w)), min(H, math.ceil(y2 + PAD * h)))


def chon_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def tao_mo_hinh(ckpt=CKPT):
    """Dựng DETR của DP-HOI đúng cấu hình pre-train (scripts/pretrain/train.sh: --dec_layers 3) và nạp trọng số."""
    args = argparse.Namespace(hidden_dim=256, position_embedding='sine', dropout=0.1, nheads=8, dim_feedforward=2048,
                              enc_layers=6, dec_layers=3, pre_norm=False)
    # Như Backbone trong models/backbone.py nhưng không tải trọng số ImageNet (đằng nào cũng ghi đè bằng checkpoint)
    resnet = torchvision.models.resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = Joiner(BackboneBase(resnet, False, 2048, False), build_position_encoding(args))
    backbone.num_channels = 2048
    model = DETR(backbone, build_transformer(args), num_classes=NUM_CLASSES, num_queries=NUM_QUERIES)
    sd = torch.load(ckpt, map_location='cpu', weights_only=False)
    model.load_state_dict(sd.get('model', sd), strict=True)
    return model.eval()


class CatNguoi:
    """ảnh PIL → danh sách (crop PIL, bbox người (x1,y1,x2,y2), điểm tin cậy), xếp theo điểm giảm dần."""

    def __init__(self, nguong=0.7, ckpt=CKPT, device=None, nms_iou=0.7, min_canh=32, min_size=MIN_SIZE, max_size=MAX_SIZE):
        self.device = device or chon_device()
        self.model = tao_mo_hinh(ckpt).to(self.device)
        self.nguong, self.nms_iou, self.min_canh = nguong, nms_iou, min_canh
        self.min_size, self.max_size = min_size, max_size     # giảm (vd 480) để chạy nhanh hơn, như với camera

    @torch.no_grad()
    def _detr(self, x):
        """Nhánh detection của DETR.forward lúc eval (prepare_for_dn khi không train: không có query nhiễu)."""
        m = self.model
        features, pos = m.backbone(nested_tensor_from_tensor_list([x]))
        src, mask = features[-1].decompose()
        bs = src.shape[0]
        tgt = m.label_enc_od(torch.tensor(m.num_classes, device=x.device)).repeat(m.num_queries, 1)
        tgt = torch.cat([tgt, torch.zeros(m.num_queries, 1, device=x.device)], dim=1)
        tgt = tgt.repeat(bs, 1, 1).transpose(0, 1)
        query = m.query_embed.weight.repeat(bs, 1, 1).transpose(0, 1)
        hs = m.transformer(m.input_proj(src), mask, 'detection', query, pos[-1], tgt=tgt, attn_mask=None)
        return m.class_embed(hs)[-1, 0], m.bbox_embed(hs).sigmoid()[-1, 0]

    @torch.no_grad()
    def tim_nguoi(self, img):
        """→ boxes (N,4) theo pixel ảnh gốc, scores (N,), chỉ lớp người, đã lọc ngưỡng + NMS."""
        img = img.convert('RGB')
        W, H = img.size
        s = self.min_size / min(W, H)
        if max(W, H) * s > self.max_size:
            s = self.max_size / max(W, H)
        x = TF.resize(img, [round(H * s), round(W * s)])
        x = TF.normalize(TF.to_tensor(x), MEAN, STD).to(self.device)
        logits, boxes = self._detr(x)
        prob = F.softmax(logits, -1)[:, :-1]                # bỏ cột "no-object"
        scores, labels = prob.max(-1)
        keep = (labels == PERSON) & (scores >= self.nguong)
        boxes = box_cxcywh_to_xyxy(boxes[keep]) * torch.tensor([W, H, W, H], device=self.device)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
        scores = scores[keep]
        du_lon = ((boxes[:, 2] - boxes[:, 0]) >= self.min_canh) & ((boxes[:, 3] - boxes[:, 1]) >= self.min_canh)
        boxes, scores = boxes[du_lon].cpu(), scores[du_lon].cpu()
        k = torchvision.ops.nms(boxes, scores, self.nms_iou)  # DETR ít khi trùng box, NMS chỉ để chắc ăn
        return boxes[k], scores[k]

    def __call__(self, img, mot_nguoi=False):
        img = img.convert('RGB')
        boxes, scores = self.tim_nguoi(img)
        if mot_nguoi and len(boxes):
            # người chính = to nhất trong ảnh, tính cả độ tin cậy
            dien_tich = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            i = int((dien_tich * scores).argmax())
            boxes, scores = boxes[i:i + 1], scores[i:i + 1]
        ket_qua = []
        for b, s in zip(boxes.tolist(), scores.tolist()):
            ket_qua.append((img.crop(khung_cat(b, *img.size)), [round(v, 1) for v in b], round(s, 4)))
        return ket_qua


def ve_khung(img, ket_qua):
    img = img.convert('RGB').copy()
    d = ImageDraw.Draw(img)
    for i, (_, b, s) in enumerate(ket_qua):
        d.rectangle(khung_cat(b, *img.size), outline=(255, 200, 0), width=2)       # khung cắt (đã nới)
        d.rectangle(b, outline=(255, 0, 0), width=3)                              # bbox người
        d.text((b[0] + 4, b[1] + 2), f'#{i} {s:.2f}', fill=(255, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('vao', nargs='+', help='ảnh hoặc thư mục ảnh')
    ap.add_argument('--out', default='crops_dphoi')
    ap.add_argument('--nguong', type=float, default=0.7, help='ngưỡng điểm lớp người (mặc định 0.7)')
    ap.add_argument('--mot-nguoi', action='store_true', help='chỉ giữ 1 người chính / ảnh')
    ap.add_argument('--ve', action='store_true', help='lưu thêm ảnh gốc có vẽ khung vào <out>/_ve/')
    ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args()

    files = []
    for p in a.vao:
        if os.path.isdir(p):
            files += sorted(os.path.join(p, f) for f in os.listdir(p) if f.lower().endswith(DUOI_ANH))
        else:
            files.append(p)
    os.makedirs(a.out, exist_ok=True)
    if a.ve:
        os.makedirs(os.path.join(a.out, '_ve'), exist_ok=True)

    cat = CatNguoi(nguong=a.nguong, device=torch.device('cpu') if a.cpu else None)
    print(f'device: {cat.device}, {len(files)} ảnh')
    with open(os.path.join(a.out, 'crops.csv'), 'w', encoding='utf-8') as f:
        f.write('crop_path,source_image,person_bbox,crop_bbox,score\n')
        for path in files:
            img = Image.open(path).convert('RGB')
            ket_qua = cat(img, mot_nguoi=a.mot_nguoi)
            ten = os.path.splitext(os.path.basename(path))[0]
            for i, (crop, b, s) in enumerate(ket_qua):
                out = os.path.join(a.out, f'{ten}_{i}.jpg')
                crop.save(out, quality=95)
                f.write(f'{os.path.relpath(out, a.out)},{path},"{",".join(map(str, b))}",'
                        f'"{",".join(map(str, khung_cat(b, *img.size)))}",{s}\n')
            if a.ve:
                ve_khung(img, ket_qua).save(os.path.join(a.out, '_ve', f'{ten}.jpg'), quality=90)
            print(f'{os.path.basename(path)}: {len(ket_qua)} người', [s for _, _, s in ket_qua])


if __name__ == '__main__':
    main()

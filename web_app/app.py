"""Web demo inference: ảnh gốc → DP-HOI tìm người + vật → cắt crop → CLIP ViT-B/32 + SVM (clip_svm.joblib) → 6 lớp.

Cách cắt: 3 quy tắc, mỗi quy tắc có một lý do
  1. Phát hiện: DP-HOI tìm người và vật (80 lớp COCO), giữ những khung có độ tin cậy >= 0.7.
     Lý do: dùng chung một ngưỡng với dphoi_detector.py, không thêm tham số mới.
  2. Gán vật cho người: vật thuộc về người nếu khung vật CHẠM khung người (hai khung giao nhau).
     Chạm nhiều người thì vật thuộc về người có tâm khung GẦN tâm vật nhất (mỗi vật chỉ thuộc 1 người).
     Không dùng "giao nhiều nhất" vì khung người đứng gần camera rất to, bao luôn vật của người đứng sau.
     Bỏ vật lớn hơn 4 lần người (bàn ăn, giường, xe buýt...): đó là cảnh nền, gộp vào thì người chỉ còn
     một góc nhỏ của crop sau khi thu về 224x224.
     Bỏ vật có khung chứa tâm của một người khác: đó là vật dùng chung (2 người 1 xe) hoặc detector gộp
     nhiều vật thành 1 khung (2 xe máy chạy cạnh nhau), không phải vật riêng của người này.
     Lý do: người tương tác với vật thì tay/thân phải chạm vật (cầm điện thoại, ngồi trên ngựa, gõ laptop).
  3. Khung cắt:
     a. Có vật: khung nhỏ nhất bao cả người và các vật của người đó, không nới (đã thấy đủ người + vật).
     b. Chỉ có người: khung người nới 20% mỗi phía (giống dataset_6class_crops), để lấy được vật mà
        detector bỏ sót (điện thoại nhỏ, cốc bị che...).

Chạy (sau khi cài theo README.md):
    python app.py                  # mở http://localhost:8000
    python app.py --port 8080 --cpu
"""
import argparse, base64, io, json, math, os, sys, threading, time, uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF
from PIL import Image, ImageDraw, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dphoi_detector as dp  # noqa: E402  (đưa third_party/DP-HOI lên sys.path)
from clip_svm import PhanLoai  # noqa: E402

NGUONG = 0.7          # quy tắc 1, giống dphoi_detector.py
VAT_TO_TOI_DA = 4.0   # quy tắc 2; quy tắc 3 nới 20% bằng dp.khung_cat (dp.PAD), giống tao_dataset_6class.py

# 91 id COCO của DETR (id 0 và các id bỏ trống = 'N/A')
COCO = ['N/A', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
        'fire hydrant', 'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
        'skateboard', 'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
        'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
        'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']


def dien_tich(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def giao(a, b):
    return dien_tich((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def chua_tam(a, b):
    """Khung a có chứa tâm khung b không."""
    x, y = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return a[0] <= x <= a[2] and a[1] <= y <= a[3]


def khoang_cach_tam(a, b):
    return ((a[0] + a[2] - b[0] - b[2]) ** 2 + (a[1] + a[3] - b[1] - b[3]) ** 2) ** 0.5 / 2


class PhatHien:
    """Dùng lại DETR của dphoi_detector nhưng giữ cả các lớp vật (dphoi_detector chỉ giữ người)."""

    def __init__(self, device=None):
        self.cat = dp.CatNguoi(nguong=NGUONG, device=device)
        self.lock = threading.Lock()

    @torch.no_grad()
    def __call__(self, img):
        W, H = img.size
        c = self.cat
        s = c.min_size / min(W, H)
        if max(W, H) * s > c.max_size:
            s = c.max_size / max(W, H)
        x = TF.normalize(TF.to_tensor(TF.resize(img, [round(H * s), round(W * s)])), dp.MEAN, dp.STD).to(c.device)
        with self.lock:
            logits, boxes = c._detr(x)
        scores, labels = F.softmax(logits, -1)[:, :-1].max(-1)
        boxes = dp.box_cxcywh_to_xyxy(boxes) * torch.tensor([W, H, W, H], device=c.device)
        boxes[:, 0::2] = boxes[:, 0::2].clamp(0, W)
        boxes[:, 1::2] = boxes[:, 1::2].clamp(0, H)
        boxes, scores, labels = boxes.cpu(), scores.cpu(), labels.cpu()
        du_lon = ((boxes[:, 2] - boxes[:, 0]) >= c.min_canh) & ((boxes[:, 3] - boxes[:, 1]) >= c.min_canh)
        keep = (scores >= NGUONG) & du_lon & torch.tensor([COCO[i] != 'N/A' for i in labels.tolist()])
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        k = torchvision.ops.batched_nms(boxes, scores, labels, c.nms_iou)
        out = [{'ten': COCO[l], 'diem': round(s, 3), 'box': [round(v, 1) for v in b]}
               for b, s, l in zip(boxes[k].tolist(), scores[k].tolist(), labels[k].tolist())]
        return [o for o in out if o['ten'] == 'person'], [o for o in out if o['ten'] != 'person']


def tao_crop(img, nguoi, vat):
    """Mỗi người → 1 crop theo quy tắc 2 và 3 (docstring đầu file). Xếp người chính (to × tin cậy) lên đầu."""
    W, H = img.size
    nguoi = sorted(nguoi, key=lambda p: -dien_tich(p['box']) * p['diem'])
    vat_cua = [[] for _ in nguoi]   # quy tắc 2
    bo_vi_to = [[] for _ in nguoi]  # vật chạm người nhưng to hơn 4 lần người: hiện ra để thấy vì sao bị bỏ
    for v in vat:
        cham = [(khoang_cach_tam(v['box'], p['box']), i) for i, p in enumerate(nguoi) if giao(v['box'], p['box']) > 0]
        if not cham:
            continue
        i = min(cham)[1]
        if dien_tich(v['box']) > VAT_TO_TOI_DA * dien_tich(nguoi[i]['box']):
            bo_vi_to[i].append({**v, 'ly_do': 'to hơn 4 lần người'})
        elif any(k != i and chua_tam(v['box'], p['box']) for k, p in enumerate(nguoi)):
            bo_vi_to[i].append({**v, 'ly_do': 'khung vật chứa người khác'})
        else:
            vat_cua[i].append(v)
    ket_qua = []
    for i, p in enumerate(nguoi):
        if vat_cua[i]:   # quy tắc 3a: đã có đủ người + vật → cắt sát khung bao, không nới
            bao = [p['box']] + [v['box'] for v in vat_cua[i]]
            khung = (max(0, math.floor(min(b[0] for b in bao))), max(0, math.floor(min(b[1] for b in bao))),
                     min(W, math.ceil(max(b[2] for b in bao))), min(H, math.ceil(max(b[3] for b in bao))))
        else:            # quy tắc 3b: chỉ có người → nới 20% để lấy được vật detector bỏ sót
            khung = dp.khung_cat(p['box'], W, H)
        ket_qua.append({'id': i, 'nguoi': p, 'vat': vat_cua[i], 'vat_bo': bo_vi_to[i], 'khung_cat': list(khung),
                        'che_do': 'người + vật' if vat_cua[i] else 'chỉ người',
                        'crop': img.crop(khung)})
    return ket_qua


def ve(img, nguoi, vat, crops):
    img = img.copy()
    d = ImageDraw.Draw(img)
    lw = max(2, round(min(img.size) / 250))
    try:
        font = ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf', max(12, lw * 6))
    except OSError:
        font = ImageFont.load_default()
    for c in crops:
        d.rectangle(c['khung_cat'], outline=(245, 180, 0), width=lw)
    for v in vat:
        d.rectangle(v['box'], outline=(30, 160, 255), width=lw)
        d.text((v['box'][0] + lw, v['box'][1] + lw), f"{v['ten']} {v['diem']:.2f}", fill=(30, 160, 255), font=font)
    for c in crops:
        b = c['nguoi']['box']
        d.rectangle(b, outline=(230, 40, 60), width=lw)
        d.text((b[0] + lw, b[1] + lw), f"#{c['id'] + 1} người {c['nguoi']['diem']:.2f}", fill=(230, 40, 60), font=font)
    return img


def b64(img, max_canh=1280):
    img = img.copy()
    img.thumbnail((max_canh, max_canh))
    buf = io.BytesIO()
    img.save(buf, 'JPEG', quality=88)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


PHIEN = {}   # id phiên → danh sách crop PIL của lần xử lý ảnh gần nhất
MAX_PHIEN = 30


def doc_anh(raw):
    img = Image.open(io.BytesIO(raw))
    return ImageOps.exif_transpose(img).convert('RGB')


class Handler(BaseHTTPRequestHandler):
    phat_hien = phan_loai = None

    def log_message(self, fmt, *a):
        sys.stderr.write('[web] ' + fmt % a + '\n')

    def _gui(self, code, body, ctype='application/json; charset=utf-8'):
        if not isinstance(body, bytes):
            body = json.dumps(body, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            with open(os.path.join(HERE, 'static', 'index.html'), 'rb') as f:
                return self._gui(200, f.read(), 'text/html; charset=utf-8')
        self._gui(404, {'loi': 'không tìm thấy'})

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get('Content-Length', 0)))
        try:
            if self.path == '/api/xu-ly':
                return self._gui(200, self.xu_ly(raw))
            if self.path == '/api/du-doan':
                return self._gui(200, self.du_doan(json.loads(raw)))
            self._gui(404, {'loi': 'không tìm thấy'})
        except Exception as e:   # trả lỗi về giao diện thay vì đứt kết nối
            import traceback
            traceback.print_exc()
            self._gui(500, {'loi': f'{type(e).__name__}: {e}'})

    def xu_ly(self, raw):
        t = time.time()
        img = doc_anh(raw)
        nguoi, vat = self.phat_hien(img)
        crops = tao_crop(img, nguoi, vat)
        pid = uuid.uuid4().hex
        PHIEN[pid] = [c['crop'] for c in crops]
        while len(PHIEN) > MAX_PHIEN:
            PHIEN.pop(next(iter(PHIEN)))
        return {'phien': pid, 'kich_thuoc': img.size, 'thoi_gian': round(time.time() - t, 2),
                'anh_ve': b64(ve(img, nguoi, vat, crops)), 'so_vat': len(vat),
                'crops': [{'id': c['id'], 'anh': b64(c['crop'], 640), 'che_do': c['che_do'],
                           'nguoi_diem': c['nguoi']['diem'], 'khung_cat': c['khung_cat'],
                           'vat': [{'ten': v['ten'], 'diem': v['diem']} for v in c['vat']],
                           'vat_bo': [{'ten': v['ten'], 'diem': v['diem'], 'ly_do': v['ly_do']}
                                      for v in c['vat_bo']]} for c in crops]}

    def du_doan(self, body):
        crops = PHIEN.get(body.get('phien'))
        if crops is None:
            raise ValueError('phiên đã hết hạn, hãy bấm "Xử lý ảnh" lại')
        if not crops:
            return {'ket_qua': [], 'thoi_gian': 0}
        t = time.time()
        return {'ket_qua': self.phan_loai(crops), 'thoi_gian': round(time.time() - t, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--cpu', action='store_true', help='chạy DP-HOI trên CPU (mặc định: MPS/CUDA nếu có)')
    a = ap.parse_args()
    print('Đang nạp DP-HOI ...', flush=True)
    Handler.phat_hien = PhatHien(torch.device('cpu') if a.cpu else None)
    print(f'  device: {Handler.phat_hien.cat.device}\nĐang nạp CLIP ViT-B/32 + SVM ...', flush=True)
    Handler.phan_loai = PhanLoai()
    print(f'Sẵn sàng: http://{a.host}:{a.port}', flush=True)
    ThreadingHTTPServer((a.host, a.port), Handler).serve_forever()


if __name__ == '__main__':
    main()

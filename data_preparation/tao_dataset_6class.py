"""Gộp 4 bộ (V-COCO, PASCAL VOC, HICO-DET, Stanford40) thành 1 thư mục ảnh crop người, 6 lớp.

Nhãn theo đúng định nghĩa trong V-COCO/Dinh_nghia_bai_toan_va_cac_lop_VCOCO.docx: phần ĐỊNH NGHĨA giữ nguyên,
phần QUY TẮC (thủ tục B0-B8) được viết lại cho từng bộ. Không đủ căn cứ xếp lớp → ignore (N5), không xuất crop.

Kết quả:  dataset_6class_crops/<lớp>/<nguồn>_<id>.jpg  + labels.csv + labels_ignore.csv   (không chia train/val/test)

Các bước (chạy lại thì bước nào xong rồi sẽ được bỏ qua):
  1. Tải ảnh gốc về _raw_cache/ (~4,8 GB): PASCAL VOC 2012, Stanford40 bản gốc, 2 shard đầu của HICO-DET, ảnh COCO.
  2. Gán nhãn từng người theo quy tắc của từng bộ, cắt theo bbox người nới 20%, lưu vào thư mục lớp.
  3. Xoá crop cũ không còn hợp lệ, ghi labels.csv và labels_ignore.csv, in bảng số lượng.
  4. Kiểm tra: file khớp labels.csv, ảnh mở được, không trùng lặp, crop V-COCO trùng với bộ crop mẫu đang có.

File đầu vào phải có sẵn (do các notebook EDA tạo ra):
  vcoco_6class_crops/vcoco_6class_crops/labels.csv   nhãn + khung cắt của V-COCO
  Summary/labels_6class_v_coco.csv                   URL ảnh COCO
  Summary/labels_6class_pascal_voc.csv               bbox + hành động gốc của từng người trong VOC

Cần: pip install pillow pyarrow

    python tao_dataset_6class.py                # chạy cả 4 bước
    python tao_dataset_6class.py --xem-mau      # thêm: lưu lưới ảnh mẫu từng (nguồn, lớp) vào xem_mau/ để soát bằng mắt
    python tao_dataset_6class.py --khong-kiem-tra
"""
import ast, csv, io, math, os, random, re, sys, tarfile, time, urllib.request, zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(ROOT, '_raw_cache')
OUT = os.path.join(ROOT, 'dataset_6class_crops')
SUMMARY = os.path.join(ROOT, 'Summary')
VCOCO_LABELS = os.path.join(ROOT, 'vcoco_6class_crops', 'vcoco_6class_crops', 'labels.csv')

PAD = 0.20           # nới bbox người 20% mỗi phía, giống bộ crop V-COCO
JPEG_QUALITY = 95
HICO_N = 3000        # số ảnh đầu của split train trên HF (giống notebook EDA)

# Tên lớp trong labels.csv → tên thư mục (giống vcoco_6class_crops)
CLASS_DIR = {'hold': 'hold', 'ride': 'ride', 'eat/drink': 'eat_drink', 'talk_on_phone': 'talk_on_phone',
             'work_on_computer': 'work_on_computer', 'unknown': 'unknown'}

# ---- Hai danh sách vật đóng của mục 2.2 (tên theo 80 loại của COCO)
VAT_CAM_TAY = {'cup', 'bottle', 'wine_glass', 'bowl', 'knife', 'fork', 'spoon', 'scissors', 'book', 'remote',
               'cell_phone', 'umbrella', 'handbag', 'backpack', 'suitcase', 'tie', 'toothbrush', 'hair_drier',
               'teddy_bear', 'vase', 'clock', 'banana', 'apple', 'orange', 'sandwich', 'hot_dog', 'pizza', 'donut',
               'cake', 'carrot', 'broccoli'}
DO_THE_THAO = {'tennis_racket', 'baseball_bat', 'baseball_glove', 'sports_ball', 'frisbee', 'surfboard',
               'snowboard', 'skis', 'skateboard', 'kite'}

COT = ['dataset', 'label', 'crop_path', 'source_image', 'person_bbox', 'crop_bbox', 'crop_w', 'crop_h',
       'original_label', 'quy_tac', 'split_goc']
COT_IGNORE = ['dataset', 'source_image', 'person_bbox', 'original_label', 'ly_do']
rows_out, rows_ignore = [], []


def doc_csv(path):
    with open(path, encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def khung_cat(bbox, W, H):
    """bbox người (x1,y1,x2,y2) → khung cắt đã nới PAD mỗi phía, không vượt ra ngoài ảnh."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (max(0, math.floor(x1 - PAD * w)), max(0, math.floor(y1 - PAD * h)),
            min(W, math.ceil(x2 + PAD * w)), min(H, math.ceil(y2 + PAD * h)))


def bo(dataset, source, person_bbox, original_label, ly_do):
    rows_ignore.append({'dataset': dataset, 'source_image': source, 'person_bbox': ','.join(map(str, person_bbox)),
                        'original_label': original_label, 'ly_do': ly_do})


def luu_crop(img, dataset, label, ten, source, person_bbox, crop_bbox, original_label, quy_tac, split_goc=''):
    """Cắt img theo crop_bbox, lưu vào OUT/<lớp>/<ten>.jpg và ghi 1 dòng labels."""
    x1, y1, x2, y2 = crop_bbox
    if x2 - x1 < 1 or y2 - y1 < 1:
        return bo(dataset, source, person_bbox, original_label, 'bbox rỗng')
    rel = f'{CLASS_DIR[label]}/{ten}.jpg'
    path = os.path.join(OUT, rel)
    if not os.path.exists(path):
        img.convert('RGB').crop(crop_bbox).save(path + '.part', 'JPEG', quality=JPEG_QUALITY)
        os.replace(path + '.part', path)                   # ghi xong mới đổi tên: bị ngắt giữa chừng không để lại ảnh hỏng
    rows_out.append({'dataset': dataset, 'label': label, 'crop_path': rel, 'source_image': source,
                     'person_bbox': ','.join(map(str, person_bbox)), 'crop_bbox': ','.join(map(str, crop_bbox)),
                     'crop_w': x2 - x1, 'crop_h': y2 - y1, 'original_label': original_label, 'quy_tac': quy_tac,
                     'split_goc': split_goc})


def so_nguyen(chuoi):
    return tuple(int(v) for v in chuoi.split(','))


# ---------------------------------------------------------------- Bước 1: tải dữ liệu gốc
# (đường dẫn trong _raw_cache, các URL thử lần lượt, kích thước đúng tính bằng byte hoặc None nếu không cố định)
_HF = 'https://huggingface.co/api/datasets/zhimeng/hico_det/parquet/default/train'
CAN_TAI = [
    ('voc/VOCtrainval_11-May-2012.tar',
     ['https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar',          # mirror, nhanh hơn nhiều
      'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'], 1999639040),
    ('stanford40/Stanford40_JPEGImages.zip', ['http://vision.stanford.edu/Datasets/Stanford40_JPEGImages.zip'], 304771808),
    ('stanford40/Stanford40_XMLAnnotations.zip', ['http://vision.stanford.edu/Datasets/Stanford40_XMLAnnotations.zip'], 3290665),
    # mỗi shard ~2933 ảnh: HICO_N = 3000 cần 2 shard đầu. Tăng HICO_N thì thêm shard (2.parquet … 12.parquet)
    ('hico/train_0.parquet', [f'{_HF}/0.parquet'], None),
    ('hico/train_1.parquet', [f'{_HF}/1.parquet'], None),
]


def tai_file(urls, path, kich_thuoc=None):
    """Tải 1 file lớn (ghi ra .part rồi mới đổi tên). Đã có file đúng kích thước thì bỏ qua."""
    if os.path.exists(path) and kich_thuoc in (None, os.path.getsize(path)):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for url in urls:
        try:
            print(f'  tải {url}', flush=True)
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=120) as r, open(path + '.part', 'wb') as f:
                while True:
                    khoi = r.read(1 << 20)
                    if not khoi:
                        break
                    f.write(khoi)
            if kich_thuoc not in (None, os.path.getsize(path + '.part')):
                raise IOError(f'tải thiếu: {os.path.getsize(path + ".part")} / {kich_thuoc} byte')
            os.replace(path + '.part', path)
            return
        except Exception as e:
            print(f'  ⚠️ lỗi: {e}')
    raise SystemExit(f'Không tải được {os.path.basename(path)}. Tải tay rồi đặt vào {path}')


def tai_du_lieu():
    print('Bước 1: tải dữ liệu gốc về _raw_cache/ (ảnh COCO của V-COCO tải ở bước 2)', flush=True)
    for rel, urls, kich_thuoc in CAN_TAI:
        tai_file(urls, os.path.join(RAW, *rel.split('/')), kich_thuoc)


# ---------------------------------------------------------------- V-COCO
THE_THAO_VCOCO = {'hit', 'kick', 'throw', 'catch', 'ski', 'surf', 'skateboard', 'snowboard'}


def nhan_vcoco(hanh_dong, vat):
    """Thủ tục B0-B8 của mục 2.2, nguyên văn. Trả về (nhãn, bước)."""
    if not hanh_dong:
        return 'ignore', 'B0'
    if 'talk_on_phone' in hanh_dong:
        return 'talk_on_phone', 'B1'
    if 'work_on_computer' in hanh_dong:
        return 'work_on_computer', 'B2'
    if hanh_dong & {'eat', 'drink'}:
        return 'eat/drink', 'B3'
    if 'ride' in hanh_dong:
        return 'ride', 'B4'
    if 'hold' not in hanh_dong:
        return 'unknown', 'B5'
    if hanh_dong & THE_THAO_VCOCO or vat in DO_THE_THAO:
        return 'unknown', 'B6'
    if vat in VAT_CAM_TAY:
        return 'hold', 'B7'
    return 'ignore', 'B8'


def tai_1_anh(url, path):
    if os.path.exists(path):
        return True
    for lan in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            with open(path + '.part', 'wb') as f:
                f.write(data)
            os.replace(path + '.part', path)
            return True
        except Exception:
            time.sleep(2 * (lan + 1))
    return False


def lam_vcoco():
    """Dùng nhãn + khung cắt của bộ crop V-COCO đang có (labels.csv), gộp cả train/val/test.
    Kiểm tra lại: nhãn trong labels.csv phải trùng với thủ tục B0-B8 áp lên nhãn gốc."""
    rows = doc_csv(VCOCO_LABELS)
    url_cua = {}
    for r in doc_csv(os.path.join(SUMMARY, 'labels_6class_v_coco.csv')):
        url_cua[int(re.search(r'_(\d{12})\.jpg$', r['image']).group(1))] = r['image']

    thu_muc = os.path.join(RAW, 'coco')
    os.makedirs(thu_muc, exist_ok=True)
    ids = sorted({int(r['image_id']) for r in rows})
    viec = [(url_cua[i], os.path.join(thu_muc, os.path.basename(url_cua[i]))) for i in ids]
    print(f'V-COCO: {len(rows)} crop từ {len(ids)} ảnh COCO, đang tải ảnh còn thiếu...', flush=True)
    with ThreadPoolExecutor(64) as ex:
        ok = list(ex.map(lambda v: tai_1_anh(*v), viec))
    loi = {i for i, o in zip(ids, ok) if not o}
    if loi:
        print(f'  ⚠️ không tải được {len(loi)} ảnh COCO')

    theo_anh = defaultdict(list)
    for r in rows:
        theo_anh[int(r['image_id'])].append(r)
    lech = 0
    for i, ds in theo_anh.items():
        img = None if i in loi else Image.open(os.path.join(thu_muc, os.path.basename(url_cua[i])))
        for r in ds:
            nhan, buoc = nhan_vcoco(set(filter(None, r['original_label'].split('+'))), r['hold_object'].replace(' ', '_'))
            if nhan != r['label']:
                lech += 1
                bo('v_coco', url_cua[i], so_nguyen(r['person_bbox']), r['original_label'],
                   f"labels.csv ghi {r['label']} nhưng B0-B8 ra {nhan}")
            elif img is None:
                bo('v_coco', url_cua[i], so_nguyen(r['person_bbox']), r['original_label'], 'không tải được ảnh')
            else:
                luu_crop(img, 'v_coco', nhan, f"vcoco_{r['image_id']}_{r['ann_id']}", url_cua[i],
                         so_nguyen(r['person_bbox']), so_nguyen(r['crop_bbox']), r['original_label'], buoc, r['split'])
    print(f'  kiểm tra B0-B8: {lech} / {len(rows)} dòng lệch nhãn')


# ---------------------------------------------------------------- PASCAL VOC
def nhan_voc(hanh_dong):
    """VOC Action gán cho mỗi người một tập con của 10 hành động (+ other). Không có hành động ăn uống,
    không có 'hold' và không có tên vật, nên lớp eat/drink và hold không lấy được từ bộ này."""
    if 'phoning' in hanh_dong:
        return 'talk_on_phone', 'B1: phoning'
    if 'usingcomputer' in hanh_dong:
        return 'work_on_computer', 'B2: usingcomputer'
    if hanh_dong & {'ridingbike', 'ridinghorse'}:
        return 'ride', 'B4: ridingbike/ridinghorse'
    for a, ly_do in (('reading', 'reading: không biết tay có cầm sách (hold) hay sách đặt trên bàn (unknown C)'),
                     ('takingphoto', 'takingphoto: cầm máy ảnh, vật ngoài danh sách vật cầm tay'),
                     ('playinginstrument', 'playinginstrument: cầm nhạc cụ, vật ngoài danh sách vật cầm tay')):
        if a in hanh_dong:
            return 'ignore', ly_do
    if hanh_dong & {'running', 'jumping'}:
        return 'unknown', 'nhóm A: running/jumping'
    if 'walking' in hanh_dong:
        return 'ignore', 'walking: VOC không ghi tay có cầm vật hay không (nhiều người xách túi, cầm đồ)'
    return 'ignore', 'other: không có hành động nào được ghi'


def lam_voc():
    """Mỗi dòng của labels_6class_pascal_voc.csv = 1 người (bbox + các hành động gốc). Đọc ảnh thẳng từ file tar."""
    theo_anh = defaultdict(list)
    for r in doc_csv(os.path.join(SUMMARY, 'labels_6class_pascal_voc.csv')):
        theo_anh[r['image']].append(r)                     # 'VOCdevkit/VOC2012/JPEGImages/xxx.jpg'
    print(f'PASCAL VOC: {sum(map(len, theo_anh.values()))} người trong {len(theo_anh)} ảnh', flush=True)
    da_thay = set()
    with tarfile.open(os.path.join(RAW, 'voc', 'VOCtrainval_11-May-2012.tar')) as tar:
        for m in tar:
            if m.name not in theo_anh:
                continue
            da_thay.add(m.name)
            img = None
            ma = os.path.splitext(os.path.basename(m.name))[0]
            for k, r in enumerate(theo_anh[m.name]):
                pb = so_nguyen(r['person_bbox'])
                nhan, quy_tac = nhan_voc(set(r['original_label'].split('+')))
                if nhan == 'ignore':
                    bo('pascal_voc', m.name, pb, r['original_label'], quy_tac)
                    continue
                if img is None:
                    img = Image.open(io.BytesIO(tar.extractfile(m).read()))
                luu_crop(img, 'pascal_voc', nhan, f'voc_{ma}_{k}', m.name, pb, khung_cat(pb, *img.size),
                         r['original_label'], quy_tac, 'trainval')
    for ten in set(theo_anh) - da_thay:
        for r in theo_anh[ten]:
            bo('pascal_voc', ten, so_nguyen(r['person_bbox']), r['original_label'], 'không thấy ảnh trong tar')


# ---------------------------------------------------------------- HICO-DET
# Con vật / phương tiện chở người (lớp 4). Ván trượt, ván lướt, ván tuyết, ski thuộc DO_THE_THAO, không tính ride.
PHUONG_TIEN = {'airplane', 'bicycle', 'boat', 'bus', 'car', 'motorcycle', 'train', 'truck',
               'horse', 'elephant', 'cow', 'sheep', 'giraffe'}
HICO_RIDE = {'ride', 'drive', 'row', 'sail', 'race', 'straddle', 'sit_on', 'stand_on'}   # đang ở trên/trong phương tiện
HICO_MAY_TINH = {('type_on', 'laptop'), ('type_on', 'keyboard'), ('read', 'laptop'), ('control', 'mouse')}
HICO_AN_UONG = {'eat', 'drink_with', 'sip', 'eat_at'}
# 'carry' balo/túi/cà vạt có thể là đeo trên người (unknown C) nên không tính là tay giữ vật
HICO_XACH = VAT_CAM_TAY - {'backpack', 'handbag', 'tie'}
# Hành động chỉ làm được khi tay đang cầm vật (vật thuộc VAT_CAM_TAY)
HICO_TAY_CAM = {('text_on', 'cell_phone'), ('read', 'cell_phone'), ('drag', 'suitcase'), ('cut_with', 'knife'),
                ('cut_with', 'scissors'), ('wield', 'knife'), ('brush_with', 'toothbrush'),
                ('operate', 'hair_drier'), ('point', 'remote'), ('toast', 'wine_glass'), ('lift', 'fork')}


def nhan_hico(cap):
    """cap: tập (động từ, vật) của 1 người, đã bỏ no_interaction. HICO-DET chỉ gán nhãn cho các loại vật được
    chọn của từng ảnh, nên người không rơi vào bước nào bên dưới là KHÔNG đủ căn cứ để nói unknown → ignore."""
    if not cap:
        return 'ignore', 'B0: chỉ có no_interaction'
    if ('talk_on', 'cell_phone') in cap:
        return 'talk_on_phone', 'B1: talk_on cell_phone'
    if cap & HICO_MAY_TINH:
        return 'work_on_computer', 'B2: type_on laptop/keyboard, read laptop, control mouse'
    if any(v in HICO_AN_UONG for v, o in cap):
        return 'eat/drink', 'B3: eat, drink_with, sip, eat_at'
    if any(v in HICO_RIDE and o in PHUONG_TIEN for v, o in cap):
        return 'ride', 'B4: ở trên/trong con vật, phương tiện'
    if any(o in DO_THE_THAO for v, o in cap):
        return 'unknown', 'B6 (nhóm B): tương tác với đồ thể thao'
    if any((v == 'hold' and o in VAT_CAM_TAY) or (v == 'carry' and o in HICO_XACH) or (v, o) in HICO_TAY_CAM
           for v, o in cap):
        return 'hold', 'B7: tay cầm vật thuộc danh sách'
    return 'ignore', 'B8: tương tác khác, không đủ căn cứ xếp lớp'


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    hop = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / hop if hop > 0 else 0


def tach_nguoi(cap):
    """HICO-DET vẽ lại bbox người cho từng cặp (vật, động từ). Gom các bbox trùng nhau (IoU ≥ 0.5) thành 1 người.
    Trả về [(bbox trung bình, {(động từ, vật)}, không_tách_được)]. không_tách_được = có bbox của người khác
    chồng lên 25-50%: không biết là 1 hay 2 người."""
    nhom = list(range(len(cap)))

    def goc(i):
        while nhom[i] != i:
            i = nhom[i]
        return i
    for i in range(len(cap)):
        for j in range(i + 1, len(cap)):
            if iou(cap[i][0], cap[j][0]) >= 0.5:
                nhom[goc(j)] = goc(i)
    theo_nhom = defaultdict(list)
    for i in range(len(cap)):
        theo_nhom[goc(i)].append(i)
    kq = []
    for g, tv in theo_nhom.items():
        box = tuple(round(sum(cap[i][0][k] for i in tv) / len(tv)) for k in range(4))
        mo_ho = any(iou(cap[i][0], cap[j][0]) >= 0.25 for i in tv for j in range(len(cap)) if goc(j) != g)
        kq.append((box, {cap[i][1] for i in tv}, mo_ho))
    return kq


def lam_hico():
    """HICO_N ảnh đầu split train của zhimeng/hico_det. Đơn vị gán nhãn là từng người (N1), không phải cả ảnh."""
    import pyarrow.parquet as pq
    print(f'HICO-DET: {HICO_N} ảnh', flush=True)
    idx = 0
    for shard in sorted(os.listdir(os.path.join(RAW, 'hico')), key=lambda s: int(re.search(r'\d+', s).group())):
        pf = pq.ParquetFile(os.path.join(RAW, 'hico', shard))
        for lo in pf.iter_batches(batch_size=64, columns=['image', 'objects', 'positive_captions', 'positive_objects']):
            for anh, objs, caps, ids in zip(*(lo.column(c).to_pylist() for c in lo.schema.names)):
                if idx >= HICO_N:
                    return
                idx += 1
                nguon = f'hico_det_train_idx_{idx - 1}'
                caps, ids = ast.literal_eval(caps), ast.literal_eval(ids)
                cap = []
                for o in ast.literal_eval(objs):
                    if o['id'] - 1 in ids and len(o['bbox_human']) == 4:       # id đánh số từ 1; vài cặp thiếu bbox người
                        obj, verb = caps[ids.index(o['id'] - 1)]
                        hb = o['bbox_human']                                   # [xmin, xmax, ymin, ymax]
                        cap.append(((hb[0], hb[2], hb[1], hb[3]), (verb, obj)))
                img = None
                for k, (pb, vo, mo_ho) in enumerate(tach_nguoi(cap)):
                    goc_ = '+'.join(sorted(f'{v} {o}' for v, o in vo))
                    nhan, quy_tac = nhan_hico({(v, o) for v, o in vo if v != 'no_interaction'})
                    if mo_ho and nhan != 'ignore':
                        nhan, quy_tac = 'ignore', 'bbox chồng 25-50% với bbox khác: không tách được người'
                    if nhan == 'ignore':
                        bo('hico_det', nguon, pb, goc_, quy_tac)
                        continue
                    if img is None:
                        img = Image.open(io.BytesIO(anh['bytes']))
                    luu_crop(img, 'hico_det', nhan, f'hico_train_{idx - 1:05d}_{k}', nguon, pb,
                             khung_cat(pb, *img.size), goc_, quy_tac, 'train')
    print(f'  ⚠️ thiếu shard parquet: mới đọc được {idx} / {HICO_N} ảnh')


# ---------------------------------------------------------------- Stanford40
# Mỗi ảnh có đúng 1 hành động (trong 40) của 1 người. Hành động không có ở đây → ignore (lý do ở ST40_BO).
ST40_LOP = {
    'phoning': ('talk_on_phone', 'B1: điện thoại áp tai'),
    'using_a_computer': ('work_on_computer', 'B2'),
    'drinking': ('eat/drink', 'B3'),
    'riding_a_bike': ('ride', 'B4'), 'riding_a_horse': ('ride', 'B4'), 'rowing_a_boat': ('ride', 'B4: ngồi trong thuyền'),
    'holding_an_umbrella': ('hold', 'B7: umbrella'),
    'texting_message': ('hold', 'B7: cell_phone cầm trước mặt, không áp tai'),
    'brushing_teeth': ('hold', 'B7: toothbrush'),
    'cutting_vegetables': ('hold', 'B7: knife, chuẩn bị đồ ăn mà không ăn'),
    'applauding': ('unknown', 'nhóm A: tay không cầm vật'), 'waving_hands': ('unknown', 'nhóm A: tay không cầm vật'),
    'jumping': ('unknown', 'nhóm A'), 'running': ('unknown', 'nhóm A'),
    'climbing': ('unknown', 'nhóm B: thể thao'), 'shooting_an_arrow': ('unknown', 'nhóm B: thể thao'),
    'throwing_frisby': ('unknown', 'nhóm B: frisbee'),
}
_VAT_LA = 'tay cầm vật ngoài danh sách vật cầm tay'
ST40_BO = {
    'blowing_bubbles': _VAT_LA, 'cleaning_the_floor': _VAT_LA, 'cutting_trees': _VAT_LA, 'fishing': _VAT_LA,
    'gardening': _VAT_LA, 'playing_guitar': _VAT_LA, 'playing_violin': _VAT_LA, 'smoking': _VAT_LA,
    'taking_photos': _VAT_LA, 'walking_the_dog': _VAT_LA, 'writing_on_a_board': _VAT_LA,
    'writing_on_a_book': _VAT_LA, 'looking_through_a_telescope': _VAT_LA,
    'looking_through_a_microscope': 'tay đặt trên kính hiển vi: không rõ có tính là giữ vật',
    'cooking': 'không biết tay cầm gì (dao, thìa thuộc danh sách; chảo, xẻng thì không)',
    'washing_dishes': 'không biết tay cầm gì (bát, cốc thuộc danh sách; đĩa thì không)',
    'pouring_liquid': 'rót, không phải ăn uống; không biết vật đang cầm (chai, cốc hay bình, ấm)',
    'feeding_a_horse': 'tay đưa đồ ăn cho ngựa: vật không rõ, không ở trên ngựa',
    'fixing_a_bike': 'sửa xe: tay giữ xe/dụng cụ, không ở trên xe',
    'fixing_a_car': 'sửa xe: tay giữ xe/dụng cụ, không ở trong xe',
    'pushing_a_cart': 'đẩy xe: tay giữ xe, không ở trên xe',
    'reading': 'không biết tay có cầm sách (hold) hay sách đặt trên bàn (unknown C), hoặc đọc báo/giấy',
    'watching_TV': 'không biết có cầm remote (hold) hay không',
}


def lam_stanford40():
    """Bản OpenML trong Summary/ (128x128) không có bbox, nên dùng bản GỐC của Stanford (có bbox người)."""
    z_anh = zipfile.ZipFile(os.path.join(RAW, 'stanford40', 'Stanford40_JPEGImages.zip'))
    z_xml = zipfile.ZipFile(os.path.join(RAW, 'stanford40', 'Stanford40_XMLAnnotations.zip'))
    ten_anh = {os.path.basename(n): n for n in z_anh.namelist() if n.lower().endswith('.jpg')}
    xmls = sorted(n for n in z_xml.namelist() if n.endswith('.xml'))
    print(f'Stanford40 (bản gốc): {len(xmls)} ảnh', flush=True)
    for n in xmls:
        root = ET.fromstring(z_xml.read(n))
        fn = root.findtext('filename')
        nguoi = root.findall('object')
        for k, o in enumerate(nguoi):
            act = o.findtext('action')
            b = o.find('bndbox')
            pb = tuple(int(float(b.findtext(t))) for t in ('xmin', 'ymin', 'xmax', 'ymax'))
            if act in ST40_BO:
                bo('stanford40', fn, pb, act, ST40_BO[act])
            elif fn not in ten_anh:
                bo('stanford40', fn, pb, act, 'không thấy ảnh')
            else:
                nhan, quy_tac = ST40_LOP[act]              # hành động lạ → KeyError, để không âm thầm bỏ sót
                img = Image.open(io.BytesIO(z_anh.read(ten_anh[fn])))
                ten = 'stanford40_' + os.path.splitext(fn)[0] + (f'_{k}' if len(nguoi) > 1 else '')
                luu_crop(img, 'stanford40', nhan, ten, fn, pb, khung_cat(pb, *img.size), act, quy_tac)


# ---------------------------------------------------------------- Bước 4: kiểm tra
def dhash(img, n=16):
    """dHash n*n bit: 2 ảnh cùng nội dung (dù nén lại) cho cùng giá trị."""
    px = img.convert('L').resize((n + 1, n), Image.BILINEAR).tobytes()
    bits = 0
    for y in range(n):
        for x in range(n):
            bits = (bits << 1) | (px[y * (n + 1) + x] > px[y * (n + 1) + x + 1])
    return bits


def kiem_tra():
    """Đọc lại labels.csv và thư mục kết quả từ đĩa, in ra từng mục kiểm tra. Trả về số lỗi."""
    from PIL import ImageChops, ImageStat
    print('\nBước 4: kiểm tra', flush=True)
    rows = doc_csv(os.path.join(OUT, 'labels.csv'))
    loi = 0

    # a) mỗi dòng labels.csv ứng với đúng 1 file và ngược lại; thư mục lớp khớp cột label
    tren_dia = {f'{d}/{f}' for d in CLASS_DIR.values() for f in os.listdir(os.path.join(OUT, d))}
    trong_csv = [r['crop_path'] for r in rows]
    thieu, thua = set(trong_csv) - tren_dia, tren_dia - set(trong_csv)
    sai_lop = sum(r['crop_path'].split('/')[0] != CLASS_DIR[r['label']] for r in rows)
    print(f'  a) {len(rows)} dòng, {len(tren_dia)} file | thiếu file {len(thieu)}, thừa file {len(thua)}, '
          f'trùng đường dẫn {len(trong_csv) - len(set(trong_csv))}, sai thư mục lớp {sai_lop}')
    loi += len(thieu) + len(thua) + sai_lop + len(trong_csv) - len(set(trong_csv))

    # b) ảnh mở được, RGB, đúng kích thước ghi trong labels.csv; c) crop trùng nhau giữa các ảnh gốc khác nhau
    hong, theo_hash = 0, defaultdict(set)
    for r in rows:
        if r['crop_path'] in thieu:
            continue
        try:
            im = Image.open(os.path.join(OUT, r['crop_path']))
            im.load()
            hong += im.mode != 'RGB' or im.size != (int(r['crop_w']), int(r['crop_h']))
            theo_hash[dhash(im)].add((r['dataset'], r['source_image']))
        except Exception:
            hong += 1
    trung = sum(1 for v in theo_hash.values() if len(v) > 1)
    print(f'  b) ảnh hỏng hoặc sai kích thước: {hong}')
    print(f'  c) nhóm crop trùng nhau (dHash 256 bit) từ ảnh gốc khác nhau: {trung}')
    loi += hong + trung

    # d) crop V-COCO ở đây phải trùng từng pixel với bộ crop V-COCO đang có (nếu thư mục đó có ảnh)
    cu = os.path.dirname(VCOCO_LABELS)
    so, khac = 0, 0
    for r in doc_csv(VCOCO_LABELS):
        p_cu = os.path.join(cu, r['crop_path'])
        p_moi = os.path.join(OUT, CLASS_DIR[r['label']], f"vcoco_{r['image_id']}_{r['ann_id']}.jpg")
        if os.path.exists(p_cu) and os.path.exists(p_moi):
            so += 1
            a, b = Image.open(p_cu).convert('RGB'), Image.open(p_moi)
            khac += a.size != b.size or max(ImageStat.Stat(ImageChops.difference(a, b)).mean) > 0
    print(f'  d) so với {so} crop V-COCO có sẵn trong vcoco_6class_crops: {khac} crop khác')
    loi += khac

    print('  ✅ không có lỗi' if loi == 0 else f'  ❌ {loi} lỗi')
    return loi


def xem_mau(so_anh=40, cot=10, o=160):
    """Lưu lưới ảnh mẫu ngẫu nhiên của từng (nguồn, lớp) vào xem_mau/ để soát bằng mắt."""
    thu_muc = os.path.join(ROOT, 'xem_mau')
    os.makedirs(thu_muc, exist_ok=True)
    rows = doc_csv(os.path.join(OUT, 'labels.csv'))
    rnd = random.Random(42)
    for ds in sorted({r['dataset'] for r in rows}):
        for lop, d in CLASS_DIR.items():
            sub = [r for r in rows if r['dataset'] == ds and r['label'] == lop]
            if not sub:
                continue
            chon = rnd.sample(sub, min(so_anh, len(sub)))
            luoi = Image.new('RGB', (cot * o, math.ceil(len(chon) / cot) * o), 'white')
            for k, r in enumerate(chon):
                im = Image.open(os.path.join(OUT, r['crop_path']))
                im.thumbnail((o - 4, o - 4))
                luoi.paste(im, ((k % cot) * o + 2, (k // cot) * o + 2))
            luoi.save(os.path.join(thu_muc, f'{ds}__{d}__{len(sub)}.jpg'), quality=88)
    print(f'\nĐã lưu lưới ảnh mẫu vào {thu_muc}')


# ---------------------------------------------------------------- main
def ghi_csv(ten, cot, rows):
    with open(os.path.join(OUT, ten), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cot)
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    tai_du_lieu()
    print('Bước 2: gán nhãn và cắt ảnh', flush=True)
    for d in CLASS_DIR.values():
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    for ham in (lam_stanford40, lam_hico, lam_voc, lam_vcoco):
        t = time.time()
        ham()
        print(f'  xong {ham.__name__} ({time.time() - t:.0f}s), tổng {len(rows_out)} crop', flush=True)

    # Xoá crop cũ không còn thuộc bộ dữ liệu (đổi lớp hoặc thành ignore sau khi sửa quy tắc)
    con_dung = {r['crop_path'] for r in rows_out}
    da_xoa = 0
    for d in CLASS_DIR.values():
        for f in os.listdir(os.path.join(OUT, d)):
            if f'{d}/{f}' not in con_dung:
                os.remove(os.path.join(OUT, d, f))
                da_xoa += 1
    ghi_csv('labels.csv', COT, rows_out)
    ghi_csv('labels_ignore.csv', COT_IGNORE, rows_ignore)

    nguon = ['v_coco', 'pascal_voc', 'hico_det', 'stanford40']
    dem = Counter((r['label'], r['dataset']) for r in rows_out)
    print('\n| lớp | ' + ' | '.join(nguon) + ' | TỔNG |\n|---|' + '---|' * (len(nguon) + 1))
    for lop in CLASS_DIR:
        so = [dem[(lop, n)] for n in nguon]
        print(f'| {lop} | ' + ' | '.join(map(str, so)) + f' | {sum(so)} |')
    so = [sum(dem[(lop, n)] for lop in CLASS_DIR) for n in nguon]
    print('| TỔNG | ' + ' | '.join(map(str, so)) + f' | {sum(so)} |')
    bo_theo_nguon = Counter(r['dataset'] for r in rows_ignore)
    print('| ignore (không xuất) | ' + ' | '.join(str(bo_theo_nguon[n]) for n in nguon) + f' | {len(rows_ignore)} |')
    print(f'\nĐã xoá {da_xoa} crop cũ không còn hợp lệ')
    print('Cạnh ngắn < 32px:', sum(1 for r in rows_out if min(r['crop_w'], r['crop_h']) < 32))
    print('\nLý do ignore:')
    for (ds, ly_do), n in sorted(Counter((r['dataset'], r['ly_do']) for r in rows_ignore).items(),
                                 key=lambda x: (nguon.index(x[0][0]), -x[1])):
        print(f'  {ds:11s} {n:5d}  {ly_do}')

    so_loi = 0 if '--khong-kiem-tra' in sys.argv else kiem_tra()
    if '--xem-mau' in sys.argv:
        xem_mau()
    sys.exit(1 if so_loi else 0)

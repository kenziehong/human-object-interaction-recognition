"""v2 của tao_dataset_6class.py: cắt lại đúng các crop của dataset_6class_crops (bộ đã lọc) theo CÁCH CẮT CỦA WEB_APP,
dùng bbox người + vật CÓ SẴN trong annotation của từng bộ (không chạy mô hình nào).

Giữ nguyên: danh sách crop, tên file, thư mục lớp, nhãn (đọc từ <vao>/labels.csv). Chỉ đổi khung cắt.

Cách cắt = quy tắc 2 và 3 của web_app/app.py (hàm tao_crop), chép nguyên sang đây, thay detector bằng annotation:
  2. Gán vật cho người: vật thuộc về người nếu khung vật CHẠM khung người; chạm nhiều người thì thuộc người có tâm
     gần tâm vật nhất. Bỏ vật: đồ nội thất lớn (DO_NEN), to hơn 4 lần người, khung vật chứa tâm của người khác.
  3. Khung cắt:  a. người có vật  → khung nhỏ nhất bao người + các vật của người đó, không nới;
                 b. chỉ có người → bbox người nới 20% mỗi phía (= khung cắt của bộ v1).

Annotation dùng cho từng nguồn:
  v_coco      người + vật: mọi annotation COCO của ảnh (instances_vcoco_all_2014.json), bỏ iscrowd.
  hico_det    người + vật: bbox người/vật của các cặp HOI dương; gom bbox trùng (IoU ≥ 0.5) như v1 (tach_nguoi).
              HICO-DET chỉ gán bbox cho vật có tương tác, nên vật khác trong ảnh không được biết.
  pascal_voc  chỉ người (VOC Action không gán bbox vật)        → luôn là quy tắc 3b.
  stanford40  chỉ người (Stanford40 chỉ có bbox người)          → luôn là quy tắc 3b.
Vật có cạnh < 32 px bị bỏ, giống bước lọc khung quá nhỏ của detector trong web_app (min_canh=32).

Cần: pip install pillow pyarrow      Dữ liệu gốc: thư mục _raw_cache/ do tao_dataset_6class.py tải về.

    python tao_dataset_6class_v2.py
    python tao_dataset_6class_v2.py --raw ~/Documents/Datasets_v2/_raw_cache \
        --vao ../data/dataset_6class_crops --ra ../data/dataset_6class_crops_v3
"""
import argparse, ast, csv, io, json, math, os, re, tarfile, time, zipfile
from collections import Counter, defaultdict

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
JPEG_QUALITY = 95

# ---- giống web_app/app.py và web_app/dphoi_detector.py
PAD = 0.20
DO_NEN = {'bed', 'couch', 'dining table', 'bench', 'toilet'}
VAT_TO_TOI_DA = 4.0
MIN_CANH = 32


def khung_cat(bbox, W, H):
    """bbox người → nới PAD mỗi phía, không vượt ra ngoài ảnh (dphoi_detector.khung_cat)."""
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    return (max(0, math.floor(x1 - PAD * w)), max(0, math.floor(y1 - PAD * h)),
            min(W, math.ceil(x2 + PAD * w)), min(H, math.ceil(y2 + PAD * h)))


def dien_tich(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def giao(a, b):
    return dien_tich((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))


def chua_tam(a, b):
    x, y = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return a[0] <= x <= a[2] and a[1] <= y <= a[3]


def khoang_cach_tam(a, b):
    return ((a[0] + a[2] - b[0] - b[2]) ** 2 + (a[1] + a[3] - b[1] - b[3]) ** 2) ** 0.5 / 2


def tao_khung(W, H, nguoi, vat):
    """= app.tao_crop, không cắt ảnh. nguoi: [box], vat: [{'ten', 'box'}].
    → mỗi người (cùng thứ tự với nguoi): (khung cắt, che_do, vật giữ, vật bỏ kèm lý do)."""
    vat = [v for v in vat if v['box'][2] - v['box'][0] >= MIN_CANH and v['box'][3] - v['box'][1] >= MIN_CANH]
    vat_cua = [[] for _ in nguoi]
    vat_bo = [[] for _ in nguoi]
    for v in vat:
        cham = [(khoang_cach_tam(v['box'], p), i) for i, p in enumerate(nguoi) if giao(v['box'], p) > 0]
        if not cham:
            continue
        i = min(cham)[1]
        if v['ten'] in DO_NEN:
            vat_bo[i].append((v, 'đồ nội thất (nền)'))
        elif dien_tich(v['box']) > VAT_TO_TOI_DA * dien_tich(nguoi[i]):
            vat_bo[i].append((v, 'to hơn 4 lần người'))
        elif any(k != i and chua_tam(v['box'], p) for k, p in enumerate(nguoi)):
            vat_bo[i].append((v, 'khung vật chứa người khác'))
        else:
            vat_cua[i].append(v)
    kq = []
    for i, p in enumerate(nguoi):
        if vat_cua[i]:
            bao = [p] + [v['box'] for v in vat_cua[i]]
            khung = (max(0, math.floor(min(b[0] for b in bao))), max(0, math.floor(min(b[1] for b in bao))),
                     min(W, math.ceil(max(b[2] for b in bao))), min(H, math.ceil(max(b[3] for b in bao))))
        else:
            khung = khung_cat(p, W, H)
        kq.append((khung, 'người + vật' if vat_cua[i] else 'chỉ người', vat_cua[i], vat_bo[i]))
    return kq


# ---------------------------------------------------------------- đọc annotation + ảnh của từng nguồn
# Mỗi hàm sinh ra (source_image, ảnh PIL, danh sách bbox người, danh sách vật, {crop_path: chỉ số người}).

def so_nguyen(s):
    return tuple(int(v) for v in s.split(','))


def iou(a, b):
    i = giao(a, b)
    hop = dien_tich(a) + dien_tich(b) - i
    return i / hop if hop > 0 else 0


def nguon_vcoco(raw, rows):
    d = json.load(open(os.path.join(raw, 'vcoco', 'instances_vcoco_all_2014.json')))
    ten_lop = {c['id']: c['name'] for c in d['categories']}
    ann_cua = defaultdict(list)
    for a in d['annotations']:
        ann_cua[a['image_id']].append(a)
    for src, ds in rows.items():
        image_id = int(re.search(r'_(\d{12})\.jpg$', src).group(1))
        nguoi, vat, ann_ids = [], [], []
        for a in ann_cua[image_id]:
            x, y, w, h = a['bbox']
            box = (x, y, x + w, y + h)
            if ten_lop[a['category_id']] == 'person':
                if not a['iscrowd'] or a['id'] in {int(r['crop_path'].rsplit('_', 1)[1][:-4]) for r in ds}:
                    nguoi.append(box)
                    ann_ids.append(a['id'])
            elif not a['iscrowd']:
                vat.append({'ten': ten_lop[a['category_id']], 'box': box})
        chi_so = {r['crop_path']: ann_ids.index(int(r['crop_path'].rsplit('_', 1)[1][:-4])) for r in ds}
        yield src, Image.open(os.path.join(raw, 'coco', os.path.basename(src))), nguoi, vat, chi_so


def gom(cap, nguong=0.5):
    """Gom bbox trùng nhau (IoU ≥ nguong) thành 1 → [(bbox trung bình, [chỉ số trong cap])], thứ tự như tach_nguoi v1."""
    nhom = list(range(len(cap)))

    def goc(i):
        while nhom[i] != i:
            i = nhom[i]
        return i
    for i in range(len(cap)):
        for j in range(i + 1, len(cap)):
            if iou(cap[i], cap[j]) >= nguong:
                nhom[goc(j)] = goc(i)
    theo_nhom = defaultdict(list)
    for i in range(len(cap)):
        theo_nhom[goc(i)].append(i)
    return [(tuple(round(sum(cap[i][k] for i in tv) / len(tv)) for k in range(4)), tv) for tv in theo_nhom.values()]


def nguon_hico(raw, rows):
    """Đọc lại parquet giống lam_hico() của v1 để thứ tự người (k trong hico_train_<idx>_<k>) trùng khớp."""
    import pyarrow.parquet as pq
    can = {int(s.rsplit('_', 1)[1]): s for s in rows}
    idx = 0
    for shard in sorted(os.listdir(os.path.join(raw, 'hico')), key=lambda s: int(re.search(r'\d+', s).group())):
        pf = pq.ParquetFile(os.path.join(raw, 'hico', shard))
        for lo in pf.iter_batches(batch_size=64, columns=['image', 'objects', 'positive_captions', 'positive_objects']):
            for anh, objs, caps, ids in zip(*(lo.column(c).to_pylist() for c in lo.schema.names)):
                idx += 1
                if idx - 1 not in can:
                    if idx > max(can):
                        return
                    continue
                src = can[idx - 1]
                caps, ids = ast.literal_eval(caps), ast.literal_eval(ids)
                hb_list, vat_tho = [], []
                for o in ast.literal_eval(objs):
                    if o['id'] - 1 in ids and len(o['bbox_human']) == 4:          # đúng điều kiện của v1
                        obj = caps[ids.index(o['id'] - 1)][0]
                        hb = o['bbox_human']                                      # [xmin, xmax, ymin, ymax]
                        hb_list.append((hb[0], hb[2], hb[1], hb[3]))
                        ob = o.get('bbox_object') or []
                        if len(ob) == 4 and not o.get('invis'):
                            vat_tho.append((obj.replace('_', ' '), (ob[0], ob[2], ob[1], ob[3])))
                nguoi = [b for b, _ in gom(hb_list)]
                vat = []                                  # cùng 1 vật lặp lại ở nhiều cặp → gom theo tên + IoU
                for ten in sorted({t for t, _ in vat_tho}):
                    for b, _ in gom([b for t, b in vat_tho if t == ten]):
                        vat.append({'ten': ten, 'box': b})
                chi_so = {r['crop_path']: int(os.path.basename(r['crop_path'])[:-4].rsplit('_', 1)[1]) for r in rows[src]}
                yield src, Image.open(io.BytesIO(anh['bytes'])), nguoi, vat, chi_so


def nguon_chi_nguoi(rows, mo_anh):
    """VOC, Stanford40: không có bbox vật → mỗi người tự cắt theo quy tắc 3b."""
    for src, img in mo_anh:
        ds = rows[src]
        yield src, img, [so_nguyen(r['person_bbox']) for r in ds], [], {r['crop_path']: i for i, r in enumerate(ds)}


def anh_voc(raw, rows):
    with tarfile.open(os.path.join(raw, 'voc', 'VOCtrainval_11-May-2012.tar')) as tar:
        for m in tar:
            if m.name in rows:
                yield m.name, Image.open(io.BytesIO(tar.extractfile(m).read()))


def anh_stanford40(raw, rows):
    z = zipfile.ZipFile(os.path.join(raw, 'stanford40', 'Stanford40_JPEGImages.zip'))
    ten = {os.path.basename(n): n for n in z.namelist() if n.lower().endswith('.jpg')}
    for src in rows:
        yield src, Image.open(io.BytesIO(z.read(ten[src])))


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--raw', default=os.path.expanduser('~/Documents/Datasets_v2/_raw_cache'))
    ap.add_argument('--vao', default=os.path.join(HERE, '..', 'data', 'dataset_6class_crops'))
    ap.add_argument('--ra', default=os.path.join(HERE, '..', 'data', 'dataset_6class_crops_v3'))
    a = ap.parse_args()

    with open(os.path.join(a.vao, 'labels.csv'), encoding='utf-8-sig', newline='') as f:
        rows_v1 = list(csv.DictReader(f))
    theo_nguon = defaultdict(lambda: defaultdict(list))        # dataset → source_image → [dòng]
    for r in rows_v1:
        theo_nguon[r['dataset']][r['source_image']].append(r)
    print(f'{len(rows_v1)} crop từ {sum(map(len, theo_nguon.values()))} ảnh gốc', flush=True)

    for d in {r['crop_path'].split('/')[0] for r in rows_v1}:
        os.makedirs(os.path.join(a.ra, d), exist_ok=True)

    nguon = {
        'v_coco': lambda rs: nguon_vcoco(a.raw, rs),
        'hico_det': lambda rs: nguon_hico(a.raw, rs),
        'pascal_voc': lambda rs: nguon_chi_nguoi(rs, anh_voc(a.raw, rs)),
        'stanford40': lambda rs: nguon_chi_nguoi(rs, anh_stanford40(a.raw, rs)),
    }
    ket_qua, da_xu_ly = {}, set()
    for ds, ham in nguon.items():
        t, n = time.time(), 0
        rs = theo_nguon[ds]
        for src, img, nguoi, vat, chi_so in ham(rs):
            da_xu_ly.add((ds, src))
            img = img.convert('RGB')
            khung = tao_khung(*img.size, nguoi, vat)
            for r in rs[src]:
                k, che_do, giu, bo = khung[chi_so[r['crop_path']]]
                pb = so_nguyen(r['person_bbox'])
                p_ng = nguoi[chi_so[r['crop_path']]]
                if max(abs(x - y) for x, y in zip(pb, p_ng)) > 1:      # bbox người phải trùng bbox của v1
                    raise SystemExit(f'{r["crop_path"]}: bbox người {p_ng} khác labels.csv {pb}')
                path = os.path.join(a.ra, r['crop_path'])
                img.crop(k).save(path + '.part', 'JPEG', quality=JPEG_QUALITY)
                os.replace(path + '.part', path)
                ket_qua[r['crop_path']] = {
                    **r, 'crop_bbox': ','.join(map(str, k)), 'crop_w': k[2] - k[0], 'crop_h': k[3] - k[1],
                    'crop_bbox_v1': r['crop_bbox'], 'che_do': che_do,
                    'vat': '|'.join(f"{v['ten']}:{','.join(str(round(x)) for x in v['box'])}" for v in giu),
                    'vat_bo': '|'.join(f"{v['ten']}:{ly_do}" for v, ly_do in bo)}
                n += 1
        print(f'  {ds}: {n} crop ({time.time() - t:.0f}s)', flush=True)

    thieu = [r['crop_path'] for r in rows_v1 if r['crop_path'] not in ket_qua]
    if thieu:
        raise SystemExit(f'❌ {len(thieu)} crop không tạo được, vd {thieu[:5]}')

    cot = list(rows_v1[0].keys()) + ['crop_bbox_v1', 'che_do', 'vat', 'vat_bo']
    with open(os.path.join(a.ra, 'labels.csv'), 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cot)
        w.writeheader()
        w.writerows(ket_qua[r['crop_path']] for r in rows_v1)
    if os.path.exists(os.path.join(a.vao, 'labels_ignore.csv')):        # người bị loại: giữ nguyên như v1
        import shutil
        shutil.copyfile(os.path.join(a.vao, 'labels_ignore.csv'), os.path.join(a.ra, 'labels_ignore.csv'))

    # ---- kiểm tra
    out = list(ket_qua.values())
    tren_dia = {f'{d}/{x}' for d in os.listdir(a.ra) if os.path.isdir(os.path.join(a.ra, d))
                for x in os.listdir(os.path.join(a.ra, d)) if x.endswith('.jpg')}
    v1 = {r['crop_path'] for r in rows_v1}
    print(f'\nKiểm tra: tên file khớp v1: {tren_dia == v1} (thừa {len(tren_dia - v1)}, thiếu {len(v1 - tren_dia)})')
    lech = sum(r['che_do'] == 'chỉ người' and r['crop_bbox'] != r['crop_bbox_v1'] for r in out)
    print(f'Crop "chỉ người" có khung khác v1: {lech}')
    dem = Counter((r['dataset'], r['che_do']) for r in out)
    print('\n| nguồn | người + vật | chỉ người |\n|---|---|---|')
    for ds in nguon:
        print(f"| {ds} | {dem[(ds, 'người + vật')]} | {dem[(ds, 'chỉ người')]} |")
    print('\nVật được giữ (top 15):', Counter(v.split(':')[0] for r in out for v in r['vat'].split('|') if v).most_common(15))
    print('Vật bị bỏ theo lý do:', Counter(v.split(':', 1)[1] for r in out for v in r['vat_bo'].split('|') if v))


if __name__ == '__main__':
    main()

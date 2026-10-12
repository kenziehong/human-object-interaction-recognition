# V-COCO: ảnh crop người, 6 lớp

Tạo bởi `tao_du_lieu_crop.py` (chạy lại là ra đúng bộ này). Mỗi ảnh = 1 người, cắt theo bbox người đã nới rộng.

## Cấu trúc

```
vcoco_6class_crops/
├── train/  <lớp>/<image_id>_<ann_id>.jpg
├── val/    <lớp>/<image_id>_<ann_id>.jpg
├── test/   <lớp>/<image_id>_<ann_id>.jpg
├── labels.csv        mỗi dòng 1 crop
└── labels_all.csv    mọi người của V-COCO, kể cả người không xuất crop (ignore / bị bỏ)
```

6 thư mục lớp: `hold`, `ride`, `eat_drink`, `talk_on_phone`, `work_on_computer`, `unknown`.
Lớp `eat/drink` có tên thư mục là `eat_drink` (tên thư mục không chứa được dấu `/`); trong `labels.csv` vẫn ghi `eat/drink`.

Đọc thẳng bằng `torchvision.datasets.ImageFolder('vcoco_6class_crops/train')`, hoặc đọc `labels.csv`.

## Số lượng

| tập | số ảnh gốc | số crop |
|---|---|---|
| train | 4509 | 6803 |
| val | 753 | 1123 |
| test | 4787 | 7261 |

| lớp | train | val | test |
|---|---|---|---|
| hold | 680 (10.0%) | 92 (8.2%) | 736 (10.1%) |
| ride | 394 (5.8%) | 64 (5.7%) | 541 (7.5%) |
| eat/drink | 640 (9.4%) | 116 (10.3%) | 578 (8.0%) |
| talk_on_phone | 309 (4.5%) | 45 (4.0%) | 285 (3.9%) |
| work_on_computer | 390 (5.7%) | 61 (5.4%) | 400 (5.5%) |
| unknown | 4390 (64.5%) | 745 (66.3%) | 4721 (65.0%) |
| TỔNG | 6803 (100.0%) | 1123 (100.0%) | 7261 (100.0%) |

`unknown` lớn hơn hẳn các lớp khác và **không bị cắt bớt**. Xử lý lúc training (sampler hoặc loss có trọng số), và đánh giá bằng macro-F1 + recall từng lớp, không dùng accuracy.

## Cách chia

- **test** = toàn bộ `vcoco_test` do tác giả V-COCO chia (ảnh COCO val2014). Giữ phân bố tự nhiên, không cân bằng lại.
- **train / val** = tách từ `vcoco_trainval` (ảnh COCO train2014), chia **theo ảnh** (mọi người trong 1 ảnh vào cùng 1 tập) và giữ tỷ lệ lớp, bằng `StratifiedGroupKFold(n_splits=7, random_state=42)`, lấy 1 phần làm val.
- Dùng **val** để chọn mô hình và siêu tham số. **test chỉ dùng để báo cáo kết quả cuối cùng.**
- Người có nhãn `ignore` (999 người, không đủ căn cứ xếp lớp) không xuất crop, ở cả 3 tập như nhau.

## Kiểm tra rò rỉ dữ liệu

- 3 tập không dùng chung `image_id` nào.
- Ảnh gần trùng: so dHash 256 bit của mọi cặp ảnh khác tập, có 30 cặp lệch <= 40 bit được đưa ra xem bằng mắt (`raw/cap_anh_nghi_gan_trung.csv`). Kết quả: bỏ 5 ảnh train/val (13 crop) vì gần trùng với ảnh ở tập khác (danh sách trong `anh_gan_trung_can_bo.txt`). Không bỏ ảnh test nào.

## Cách cắt

- Khung cắt = bbox người của COCO, nới thêm **20% mỗi phía** (trái/phải theo chiều rộng, trên/dưới theo chiều cao), không vượt ra ngoài ảnh. Không resize, không đệm viền. JPEG chất lượng 95.
- Mọi người được nới **cùng một tỷ lệ**, không dùng bbox vật để cắt (nếu dùng thì hình dạng crop sẽ để lộ nhãn, và lúc dùng thật không có bbox vật).
- `labels.csv` có `person_bbox` (bbox gốc) và `crop_bbox` (khung đã cắt), đều là `x1,y1,x2,y2` trên ảnh gốc.
- Có 16 crop có cạnh ngắn < 32 pixel. Không lọc sẵn; muốn lọc thì dùng cột `crop_w`, `crop_h`.

Crop chứa được bao nhiêu vật đang tương tác, theo mức nới. Mỗi ô: trung bình % diện tích vật nằm trong crop | % số người có vật nằm >= 90% trong crop:

| lớp | số người có bbox vật | nới 0% | nới 10% | nới 20% | nới 30% |
|---|---|---|---|---|---|
| talk_on_phone | 538 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 100% |
| work_on_computer | 759 | 54% | 24% | 65% | 33% | 74% | 43% | 81% | 55% |
| eat/drink | 911 | 83% | 71% | 89% | 81% | 93% | 87% | 95% | 90% |
| ride | 924 | 31% | 3% | 39% | 9% | 47% | 14% | 53% | 20% |
| hold | 1512 | 71% | 48% | 83% | 64% | 89% | 75% | 93% | 82% |

## Lưu ý

- Nhãn kế thừa từ người gán nhãn V-COCO theo quy tắc ở Phần 5 của `V_COCO_EDA_simple_V2.ipynb`, không gán lại bằng mắt.
- Ảnh test thuộc COCO val2014. Backbone/detector pretrain trên **COCO train2017** đã thấy phần lớn các ảnh này; backbone pretrain ImageNet thì không.

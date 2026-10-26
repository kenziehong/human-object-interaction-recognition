# Chuẩn bị dữ liệu: `tao_dataset_6class.py`

Gộp 4 bộ dữ liệu (V-COCO, PASCAL VOC 2012 Action, HICO-DET, Stanford40) thành **1 thư mục ảnh crop người, 6 lớp**: `hold`, `ride`, `eat/drink`, `talk_on_phone`, `work_on_computer`, `unknown`. Mỗi ảnh là 1 người, cắt theo bbox người đã nới rộng. **Không chia train/val/test.**

Nhãn theo `Dinh_nghia_bai_toan_va_cac_lop_VCOCO.docx`: phần định nghĩa giữ nguyên, phần quy tắc (thủ tục B0–B8) viết lại cho từng nguồn. Không đủ căn cứ xếp lớp thì là `ignore` (N5): không xuất crop.

## Cài đặt

```bash
pip install pillow pyarrow
```

Cần khoảng **5 GB** trống cho ảnh gốc tải về (`_raw_cache/`) và có mạng ở lần chạy đầu.

## Chuẩn bị thư mục

Script tìm mọi file **cạnh chính nó** (`ROOT` = thư mục chứa file `.py`). Đặt script vào thư mục có sẵn các file đầu vào sau (do các notebook EDA tạo ra):

```
<thư mục làm việc>/
├── tao_dataset_6class.py
├── vcoco_6class_crops/vcoco_6class_crops/labels.csv   nhãn + khung cắt của V-COCO
└── Summary/
    ├── labels_6class_v_coco.csv                       URL ảnh COCO
    └── labels_6class_pascal_voc.csv                   bbox + hành động gốc của từng người trong VOC
```

Repo này **không** chứa các file đầu vào trên. Trên máy hiện tại, thư mục làm việc là `~/Documents/Datasets_v2/` (đã có sẵn đầu vào và `_raw_cache/`).

## Chạy

```bash
cd <thư mục làm việc>
python tao_dataset_6class.py                  # chạy cả 4 bước
python tao_dataset_6class.py --xem-mau        # thêm: lưu lưới ảnh mẫu từng (nguồn, lớp) vào xem_mau/ để soát bằng mắt
python tao_dataset_6class.py --khong-kiem-tra # bỏ bước 4 (kiểm tra)
```

Các bước (chạy lại thì bước nào xong rồi sẽ được bỏ qua, ví dụ ảnh đã tải thì không tải lại):

1. **Tải ảnh gốc** về `_raw_cache/` (~4,8 GB): PASCAL VOC 2012, Stanford40 bản gốc, 2 shard đầu của HICO-DET (3.000 ảnh, trên Hugging Face), ảnh COCO cần cho V-COCO.
2. **Gán nhãn và cắt ảnh**: gán nhãn từng người theo quy tắc của từng nguồn, cắt theo bbox người nới 20% mỗi phía, lưu vào thư mục lớp.
3. **Ghi kết quả**: xoá crop cũ không còn hợp lệ, ghi `labels.csv` và `labels_ignore.csv`, in bảng số lượng theo (lớp, nguồn) và lý do ignore.
4. **Kiểm tra**: mỗi dòng `labels.csv` khớp đúng 1 file, ảnh mở được và đúng kích thước, không có crop trùng nhau, crop V-COCO trùng với bộ crop mẫu `vcoco_6class_crops`. Có lỗi thì script thoát với mã 1.

## Kết quả

```
dataset_6class_crops/
├── hold/  ride/  eat_drink/  talk_on_phone/  work_on_computer/  unknown/
│       <nguồn>_<id>.jpg
├── labels.csv           mỗi dòng 1 crop (cột quy_tac: bước nào quyết định nhãn)
└── labels_ignore.csv    mỗi dòng 1 người bị loại, kèm lý do (cột ly_do)
```

- Tiền tố tên file cho biết nguồn: `vcoco_<image_id>_<ann_id>`, `voc_<mã ảnh>_<thứ tự người>`, `hico_train_<idx>_<thứ tự người>`, `stanford40_<hành động>_<số>`.
- Thư mục `eat_drink/` ứng với nhãn `eat/drink` trong `labels.csv`.
- Đọc bằng `torchvision.datasets.ImageFolder('dataset_6class_crops')` hoặc đọc `labels.csv`.

Chạy script cho ra bộ **trước khi lọc tay** (24.408 crop). Bộ đang dùng (22.366 crop) đã bỏ thêm 2.042 crop mờ, quá nhỏ, sai nhãn, trùng lặp; chi tiết ở `dataset_6class_crops/README.md` trong thư mục làm việc.

## Cách cắt ảnh

Hàm `khung_cat` (dòng 65–70): với bbox người `(x1, y1, x2, y2)`, `w = x2 - x1`, `h = y2 - y1`, khung cắt là

```
(x1 - 0.2·w,  y1 - 0.2·h,  x2 + 0.2·w,  y2 + 0.2·h)
```

tức crop rộng/cao gấp 1,4 lần bbox. Phía nào vượt ra ngoài ảnh thì bị cắt lại ở mép ảnh (`max(0, …)`, `min(W/H, …)`), các phía khác vẫn nới bình thường; không độn viền. V-COCO không qua hàm này mà lấy khung cắt có sẵn trong `vcoco_6class_crops/.../labels.csv` (cũng nới 20%).

## Tham số chỉnh trong file

| biến | mặc định | ý nghĩa |
|---|---|---|
| `PAD` | `0.20` | tỉ lệ nới bbox người mỗi phía |
| `JPEG_QUALITY` | `95` | chất lượng JPEG khi lưu crop |
| `HICO_N` | `3000` | số ảnh đầu của split train HICO-DET; tăng thì phải thêm shard vào `CAN_TAI` (mỗi shard ~2.933 ảnh) |

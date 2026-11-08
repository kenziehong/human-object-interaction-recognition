# Chuẩn bị dữ liệu: `tao_dataset_6class.py`

Cắt lại bộ ảnh crop người 6 lớp (`hold`, `ride`, `eat/drink`, `talk_on_phone`, `work_on_computer`, `unknown`) **theo cách cắt "người + vật"**: người có vật tương tác thì khung bao cả người lẫn vật, người không có vật thì cắt quanh người.

- **Crop**: một ảnh nhỏ cắt ra từ ảnh gốc, chứa một người (và vật người đó tương tác, nếu có).
- **bbox** (bounding box, khung chữ nhật bao quanh một đối tượng): viết `(x1, y1, x2, y2)`, tức góc trên-trái và góc dưới-phải, tính bằng pixel.

Script **không chạy mô hình nhận diện nào**. Nó dùng bbox người và bbox vật có sẵn trong annotation (nhãn do người gán) của từng bộ dữ liệu.

## Giữ nguyên và đổi gì

| | |
|---|---|
| **Giữ nguyên** | danh sách crop, tên file, thư mục lớp, nhãn (đọc từ `<vao>/labels.csv`) |
| **Chỉ đổi** | khung cắt của mỗi crop |

Ví dụ: crop `ride/vcoco_123_456.jpg` vẫn tên đó, vẫn nhãn `ride`. Khác là trước đây nó chỉ là người nới 20%, nay nếu người đang cưỡi xe đạp thì khung rộng ra để chứa cả chiếc xe.

## Cài đặt

```bash
pip install pillow pyarrow
```

Cần thư mục ảnh gốc `_raw_cache/` (khoảng 5 GB: ảnh COCO, PASCAL VOC 2012, Stanford40, HICO-DET) và bộ crop đã lọc tay `dataset_6class_crops/` làm đầu vào. Trên máy hiện tại cả hai nằm trong `~/Documents/Datasets_v2/`. Script này **không tự tải** ảnh gốc.

## Chạy

```bash
python tao_dataset_6class.py
python tao_dataset_6class.py --raw ~/Documents/Datasets_v2/_raw_cache \
    --vao ~/Documents/Datasets_v2/dataset_6class_crops \
    --ra  ../data/dataset_6class_crops_v3
```

| tham số | mặc định | ý nghĩa |
|---|---|---|
| `--raw` | `~/Documents/Datasets_v2/_raw_cache` | thư mục ảnh gốc |
| `--vao` | `../data/dataset_6class_crops` | bộ crop đầu vào (cần `labels.csv`, tuỳ chọn `labels_ignore.csv`) |
| `--ra` | `../data/dataset_6class_crops_v3` | thư mục ghi kết quả |

Chạy xong in: số crop mỗi nguồn, kiểm tra tên file khớp bộ đầu vào, bảng số crop "người + vật" / "chỉ người", các vật được giữ nhiều nhất và lý do vật bị bỏ.

## Cách cắt ảnh

**Bước 1. Lọc vật.** Vật có cạnh ngắn hơn 32 px bị bỏ (`MIN_CANH`), vì khung quá nhỏ thì không đáng tin. Ví dụ: một chiếc cốc cao 20 px trong ảnh không được tính.

**Bước 2. Gán vật cho người.** Vật thuộc về người nào có bbox **chạm** bbox vật. Nếu chạm nhiều người thì thuộc người có tâm gần tâm vật nhất. Vật bị bỏ nếu rơi vào một trong ba trường hợp:

| lý do bỏ (cột `vat_bo`) | điều kiện | ví dụ |
|---|---|---|
| đồ nội thất (nền) | tên vật thuộc `DO_NEN` = giường, ghế sofa, bàn ăn, ghế dài, bồn cầu | người ngồi trên sofa thì không lấy cả sofa |
| to hơn 4 lần người | diện tích vật > `VAT_TO_TOI_DA` (4) × diện tích người | chiếc xe buýt rất lớn so với người đứng cạnh |
| khung vật chứa người khác | tâm của một người khác nằm trong bbox vật | bbox "bàn" bao luôn người thứ hai |

**Bước 3. Chọn khung cắt.**

| chế độ (cột `che_do`) | khi nào | khung cắt |
|---|---|---|
| `người + vật` | người có ít nhất 1 vật được giữ | khung nhỏ nhất bao người và mọi vật của người đó, **không nới** |
| `chỉ người` | người không có vật | bbox người nới 20% mỗi phía (hàm `khung_cat`) |

Với `chỉ người`: bbox `(x1, y1, x2, y2)`, `w = x2 - x1`, `h = y2 - y1`, khung là `(x1 - 0.2·w, y1 - 0.2·h, x2 + 0.2·w, y2 + 0.2·h)`, tức rộng và cao gấp 1,4 lần bbox. Phía nào vượt ra ngoài ảnh thì bị cắt ở mép ảnh, không độn viền. Đây cũng là khung của bộ v1 cũ.

## Annotation dùng cho từng nguồn

| nguồn | người | vật |
|---|---|---|
| `v_coco` | mọi annotation `person` của ảnh trong `instances_vcoco_all_2014.json` | mọi annotation không phải người, bỏ `iscrowd` (nhãn đám đông) |
| `hico_det` | bbox người của các cặp tương tác dương, gom bbox trùng (IoU ≥ 0.5) | bbox vật của các cặp đó, gom theo tên + IoU |
| `pascal_voc` | bbox người | **không có** → luôn là `chỉ người` |
| `stanford40` | bbox người | **không có** → luôn là `chỉ người` |

- **IoU** (Intersection over Union): diện tích phần giao chia diện tích phần hợp của hai bbox. Hai bbox trùng hẳn thì IoU = 1, không chạm nhau thì IoU = 0.
- HICO-DET chỉ cho bbox của vật **có tương tác**, nên vật khác trong ảnh không được biết, và crop HICO-DET có thể thiếu vật so với V-COCO.
- Script dừng với lỗi nếu bbox người trong annotation lệch bbox trong `labels.csv` quá 1 px, để chắc rằng cùng một người được cắt lại.

## Kết quả

```
dataset_6class_crops_v3/
├── hold/  ride/  eat_drink/  talk_on_phone/  work_on_computer/  unknown/
│       <nguồn>_<id>.jpg
├── labels.csv           mỗi dòng 1 crop
└── labels_ignore.csv    chép nguyên từ bộ đầu vào (người đã bị loại)
```

`labels.csv` giữ mọi cột của bộ đầu vào, cột `crop_bbox` được thay bằng khung mới, và thêm 4 cột:

| cột | nội dung |
|---|---|
| `crop_bbox_v1` | khung cắt cũ (người nới 20%) để so sánh |
| `che_do` | `người + vật` hoặc `chỉ người` |
| `vat` | vật được giữ, dạng `tên:x1,y1,x2,y2`, nhiều vật ngăn bằng `\|` |
| `vat_bo` | vật bị bỏ kèm lý do, dạng `tên:lý do` |

Thư mục `eat_drink/` ứng với nhãn `eat/drink`. Đọc bằng `torchvision.datasets.ImageFolder` hoặc đọc `labels.csv`.

Bộ đã lọc tay có **22.366 crop**:

| nguồn | số crop |
|---|---|
| v_coco | 14.090 |
| stanford40 | 4.103 |
| pascal_voc | 2.753 |
| hico_det | 1.420 |

| lớp | số crop |
|---|---|
| unknown | 12.533 |
| ride | 2.864 |
| hold | 2.701 |
| eat/drink | 1.486 |
| work_on_computer | 1.455 |
| talk_on_phone | 1.327 |

Trong đó 8.927 crop là `người + vật` và 13.439 crop là `chỉ người`.

## Tham số chỉnh trong file

| biến | mặc định | ý nghĩa |
|---|---|---|
| `PAD` | `0.20` | tỉ lệ nới bbox người mỗi phía (chế độ `chỉ người`) |
| `DO_NEN` | giường, sofa, bàn ăn, ghế dài, bồn cầu | vật nền luôn bị bỏ |
| `VAT_TO_TOI_DA` | `4.0` | vật lớn hơn người bao nhiêu lần thì bị bỏ |
| `MIN_CANH` | `32` | cạnh ngắn nhất (px) của vật được giữ |
| `JPEG_QUALITY` | `95` | chất lượng JPEG khi lưu crop |

`PAD`, `DO_NEN`, `VAT_TO_TOI_DA`, `MIN_CANH` là cách cắt gốc; `web_app` dùng cùng giá trị khi chạy thật. Đổi ở đây thì đổi cả `web_app/app.py` và `web_app/dphoi_detector.py`.

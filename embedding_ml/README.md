# Embedding ảnh và phân loại

## Trích xuất embedding

Chạy từ thư mục gốc của project:

```bash
python embedding_ml/extract_embedding.py \
  --data-dir data/vcoco_6class_crops \
  --output-dir data/embedding_csv
```

Dataset cần có cấu trúc `train/<nhãn>/`, `val/<nhãn>/` và `test/<nhãn>/`. Mặc định script trích xuất ResNet50, EfficientNet-B0 và ViT-B/16, sau đó lưu một dòng cho mỗi ảnh trong `data/embedding_csv/embeddings.csv`. Ba cột embedding chứa vector JSON.

Có thể chọn model, batch size hoặc thiết bị, ví dụ:

```bash
python embedding_ml/extract_embedding.py --models resnet50 vit_b_16 --batch-size 16 --device cpu
```

Thiết bị hỗ trợ: `auto`, `mps`, `cuda`, `cpu`. `auto` tự chọn MPS, CUDA hoặc CPU.

## Chọn mô hình và đánh giá cuối

CSV cần có các cột `dataset`, `label_id` và `resnet50_embedding`,
`efficientnet_b0_embedding`, `vit_b_16_embedding`. Mỗi cột embedding là một
vector JSON. `train_search.py` chạy 12 `RandomizedSearchCV` độc lập (3
embedding × 4 classifier). Mỗi search fit candidate trên train và chọn tham số
bằng macro-F1 trên val (`PredefinedSplit`). Cấu hình tốt nhất của mỗi trường
hợp được fit trên train để ghi báo cáo train/val. Sau khi hoàn tất cả search,
12 model tốt nhất được fit lại trên train+val và lưu trong
`embedding_ml/outputs/models/`. Cuối cùng, cấu hình đứng đầu theo macro-F1 val
được đánh giá test một lần. Toàn bộ báo cáo, gồm kết quả search, test và
classification report, chỉ lưu trong `embedding_ml/outputs/all_results.txt`.

```bash
python embedding_ml/train_search.py
```

`train_search.py` chỉ dùng val để chọn cấu hình; test chỉ được đọc ở bước cuối
sau khi đã chọn cấu hình tốt nhất trong cả 12 search. Có thể đổi số lần lấy
mẫu, đường dẫn CSV và thư mục output qua các tham số CLI; xem `--help`.
# Trích embedding và chia train/val/test

Với dữ liệu hiện tại trong `data/dataset_6class_crops_v2/` (mỗi nhãn là một
folder), trích embedding trước khi gán split:

```bash
python embedding_ml/extract_embedding.py \
  --data-dir data/dataset_6class_crops_v2 \
  --output-dir data/embedding_csv \
  --batch-size 32
```

Mặc định trích ResNet50, EfficientNet-B0 và ViT-B/16; kết quả nằm trong
`data/embedding_csv/embeddings.csv`, theo định dạng các cột embedding JSON như
CSV hiện có. Có thể chỉ trích một backbone bằng `--models resnet50`.

Sau đó chia phân tầng theo nhãn, truyền seed và ba tỉ lệ theo thứ tự train, val,
test:

```bash
python embedding_ml/split.py \
  --input data/embedding_csv/embeddings.csv \
  --ratios 0.7 0.15 0.15 \
  --seed 42
```

Mặc định kết quả được ghi vào `data/embedding_csv/splits/`: `train.csv`,
`val.csv`, `test.csv` và `embeddings_split.csv` (toàn bộ dòng, có cột `dataset`
được gán). Tổng tỉ lệ được chuẩn hóa nếu không đúng 1. Việc phân bổ số mẫu mỗi
lớp dùng phương pháp phần dư lớn nhất để tổng số mẫu của các split khớp chính
xác với dữ liệu đầu vào.


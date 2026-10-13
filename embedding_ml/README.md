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
python embedding_ml/train_search.py --n-iter 30
```

`train_search.py` chỉ dùng val để chọn cấu hình; test chỉ được đọc ở bước cuối
sau khi đã chọn cấu hình tốt nhất trong cả 12 search. Có thể đổi số lần lấy
mẫu, đường dẫn CSV và thư mục output qua các tham số CLI; xem `--help`.

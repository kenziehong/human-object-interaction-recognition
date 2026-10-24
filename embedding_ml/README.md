# Image embedding classification

Hướng dẫn này mô tả luồng hiện tại: trích embedding từ ảnh, tìm classifier tốt
nhất trên validation, đánh giá trên test, rồi dùng model đã lưu để dự đoán ảnh.
Các lệnh bên dưới chạy từ thư mục gốc của project.

## 1. Chuẩn bị dữ liệu và trích embedding

Mặc định, `extract_embedding.py` đọc danh sách ảnh từ
`data/train_test_split/labels_split_unk3000.csv`. CSV cần có ba cột:

- `crop_path`: đường dẫn ảnh tương đối với thư mục `--data-dir`.
- `label`: tên lớp.
- `split`: một trong `train`, `val`, `test`.

Mặc định `--data-dir` là `data/dataset_6class_crops_v2`. Ví dụ, nếu
`crop_path` là `unknown/example.jpg`, ảnh cần nằm tại
`data/dataset_6class_crops_v2/unknown/example.jpg`.

Chạy trích xuất cả ba backbone:

```bash
python embedding_ml/extract_embedding.py \
  --data-dir data/dataset_6class_crops_v2 \
  --labels-csv data/train_test_split/labels_split_unk3000.csv \
  --output-dir data/embedding_csv \
  --batch-size 32 \
  --device auto
```

Kết quả là `data/embedding_csv/embeddings_v2.csv`, mỗi dòng ứng với một ảnh.
CSV có metadata ảnh, nhãn, split và ba cột vector JSON:
`resnet50_embedding`, `efficientnet_b0_embedding`, `vit_b_16_embedding`.
Lần chạy đầu có thể cần tải pretrained weights của torchvision. `auto` chọn
MPS, CUDA hoặc CPU; có thể chỉ định thiết bị bằng `--device`. Có thể giảm batch
nếu thiếu bộ nhớ hoặc chỉ trích một số backbone:

```bash
python embedding_ml/extract_embedding.py --models resnet50 vit_b_16 --batch-size 16 --device cpu
```

`--preprocessing weights` là mặc định và dùng transform tương ứng với weights
của mỗi backbone. Giữ nguyên thiết lập này khi chạy dự đoán để preprocessing
khớp với embedding dùng lúc train.

## 2. Train, chọn model và xem kết quả

Chạy tìm tham số cho ba backbone và bốn classifier (12 tổ hợp):

```bash
python embedding_ml/train_search.py
```

Mặc định script đọc `data/embedding_csv/embeddings_v2.csv`. Mỗi search fit
các cấu hình trên train và chọn cấu hình theo macro-F1 trên validation. Sau
khi chọn tham số, script đánh giá validation bằng model fit trên train, rồi
fit model dùng để lưu trên train+val và đánh giá model đó trên test. Test
không tham gia chọn tham số.

Các file được tạo trong `embedding_ml/outputs/`:

- `models_v2/<backbone>_<classifier>.joblib`: 12 model và metadata gồm tên
  backbone, classifier, nhãn và tham số đã chọn.
- `metrics.csv` và `metrics.json`: accuracy cùng macro/weighted precision,
  recall và F1 cho validation và test của từng tổ hợp.
- `all_results_v2.txt`: tham số tốt nhất, metric tổng hợp và classification
  report theo từng lớp cho validation và test.

Có thể đổi CSV đầu vào, thư mục output, số lần lấy mẫu ngẫu nhiên và seed:

```bash
python embedding_ml/train_search.py \
  --csv data/embedding_csv/embeddings_v2.csv \
  --output-dir embedding_ml/outputs \
  --n-iter 30 \
  --seed 42
```

## 3. Dự đoán một hoặc nhiều ảnh

Chọn file model cụ thể, sau đó truyền một hoặc nhiều ảnh:

```bash
python embedding_ml/predict.py \
  embedding_ml/outputs/models_v2/resnet50_logreg.joblib \
  path/to/image.jpg path/to/another_image.jpg
```

Mỗi ảnh cho một dòng JSON gồm `image_path`, `label_id`, `label`, `confidence`
và `model_name`. Thứ tự kết quả khớp với thứ tự ảnh đầu vào. Có thể đổi thiết
bị bằng `--device cpu|cuda|mps|auto` và kích thước batch bằng `--batch-size`.

Dùng từ Python cũng trả về một chuỗi JSON Lines:

```python
from embedding_ml.predict import predict

result = predict(
    "embedding_ml/outputs/models_v2/resnet50_logreg.joblib",
    ["path/to/image.jpg", "path/to/another_image.jpg"],
)
print(result)
```

Tham số ảnh cũng có thể là một chuỗi đơn nếu chỉ dự đoán một ảnh.

Logistic regression, random forest và XGBoost cung cấp confidence từ
`predict_proba`. SVM hiện được train không bật hiệu chuẩn xác suất; confidence
của SVM được tính bằng softmax trên decision scores và nên hiểu là điểm tin
cậy tương đối, không phải xác suất đã hiệu chuẩn.

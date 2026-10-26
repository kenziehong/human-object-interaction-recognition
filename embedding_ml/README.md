# Pipeline phân loại bằng image embedding

Chạy các lệnh từ thư mục gốc của project. Pipeline gồm ba bước: tạo embedding,
tìm classifier bằng train/validation và dự đoán ảnh mới.

## 1. Trích xuất đặc trưng (Embedding)

Ảnh crop được chia theo manifest cố định
`data/train_test_split/labels_split_unk3000.csv` thành các tập `train`, `val`
và `test`, với sáu lớp tương tác người–vật. Tiền xử lý đầu vào giống pipeline
deep learning: áp dụng hướng EXIF, chuyển ảnh sang RGB, resize giữ nguyên tỷ lệ,
padding giữa nền đen thành ảnh vuông `224×224`, rồi chuẩn hóa theo mean và độ
lệch chuẩn ImageNet. Không áp dụng tăng cường dữ liệu.

Dùng hai CNN pretrained trên ImageNet làm bộ trích xuất đặc trưng. Trọng số được
giữ nguyên và đóng băng; lớp phân loại cuối được thay bằng phép đồng nhất để lấy
vector sau global average pooling. Vì vậy backbone chỉ dùng để suy luận đặc
trưng, không được huấn luyện lại.

| Mô hình trích xuất | Đặc trưng | Số chiều | Thông tin biểu diễn |
| --- | --- | ---: | --- |
| ResNet50 | Vector sau global average pooling (`layer4`/`avgpool`) | 2.048 | Đặc trưng ngữ nghĩa mức cao, tổng hợp thông tin hình dạng, đường viền và kết cấu từ các tầng trước. |
| EfficientNet-B0 | Vector sau global average pooling (đầu ra `features`/`avgpool`, trước classifier) | 1.280 | Đặc trưng ngữ nghĩa mức cao với vector gọn; kiến trúc MBConv và compound scaling giúp cân bằng hiệu quả tính toán. |

Các vector được lưu trong CSV cùng nhãn, `label_id` và tập dữ liệu tương ứng
(`train`, `val`, `test`). Embedding là đầu vào cho các mô hình học máy ở bước
tiếp theo.

Đầu vào mặc định:

- CSV split: `data/train_test_split/labels_split_unk3000.csv`, gồm các cột
  `crop_path`, `label`, `split` (`train`, `val`, `test`).
- Thư mục ảnh: `data/dataset_6class_crops_v2/`. `crop_path` là đường dẫn tương
  đối bên trong thư mục này; ví dụ `unknown/example.jpg`.

Chạy trích xuất cho cả ResNet50 và EfficientNet-B0:

```bash
python embedding_ml/extract_embedding.py \
  --data-dir data/dataset_6class_crops_v2 \
  --labels-csv data/train_test_split/labels_split_unk3000.csv \
  --output-dir data/embedding_csv \
  --batch-size 32 \
  --device auto
```

Kết quả được ghi vào `data/embedding_csv/embeddings_v3.csv`. Mỗi ảnh có vector
`resnet50_embedding` và `efficientnet_b0_embedding`. Dùng `--device cpu` nếu
muốn chạy trên CPU; giảm `--batch-size` nếu thiếu bộ nhớ. Lần chạy đầu có thể
cần tải ImageNet weights của torchvision.

Việc trích embedding và dự đoán dùng cùng transform từ
`deep_learning/data.py`, đảm bảo tiền xử lý nhất quán.

## 2. Tìm và lưu classifier

Sau khi tạo CSV embedding, chạy:

```bash
python embedding_ml/train_search.py \
  --csv data/embedding_csv/embeddings_v3.csv \
  --model-dir embedding_ml/outputs/models_v3 \
  --results-file embedding_ml/outputs/all_results_v3.txt \
  --n-iter 12 \
  --seed 42
```

Script tìm cấu hình cho hai backbone và bốn classifier. Mỗi cấu hình được fit
trên train; chọn cấu hình có macro-F1 cao nhất trên validation. Test chỉ dùng
để báo cáo kết quả cuối, không dùng để chọn cấu hình. Model và kết quả được lưu
trong hai đường dẫn đã chỉ định. Dùng thư mục/tên file v3 để giữ riêng
kết quả của lần chạy với preprocessing này; script bỏ qua model nếu file đó đã
tồn tại.

## 3. Dự đoán ảnh

Chọn một file model đã lưu và truyền một hoặc nhiều ảnh:

```bash
python embedding_ml/predict.py \
  embedding_ml/outputs/models_v3/resnet50_logreg.joblib \
  path/to/image.jpg path/to/another_image.jpg \
  --device auto \
  --batch-size 16
```

Kết quả in ra JSON Lines theo thứ tự ảnh đầu vào, gồm `image_path`, `label_id`,
`label`, `confidence` và `model_name`. Với SVM, `confidence` được tính từ
decision scores và không phải xác suất đã hiệu chuẩn.

## 4. Ma trận nhầm lẫn trên test

Để đánh giá model ResNet50 + XGBoost bằng embedding đã lưu (không trích xuất lại
đặc trưng từ ảnh), chạy từ thư mục gốc project:

```bash
python embedding_ml/confusion_matrix.py
```

Mặc định script đọc `embedding_ml/outputs/models_v3/resnet50_xgboost.joblib`,
lấy các dòng `test` và cột `resnet50_embedding` từ
`data/embedding_csv/embeddings_v3.csv`, rồi lưu hình tại
`embedding_ml/outputs/confusion_matrix_resnet50_xgboost_test.png`. Báo cáo
precision/recall/F1 và ma trận số cũng được in ra terminal. Có thể truyền đường
dẫn khác bằng `--model`, `--csv` và `--output`; thêm `--normalize true` để hiển
thị tỷ lệ theo từng nhãn thật.

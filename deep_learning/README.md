# Phân loại ảnh tương tác người–vật

Pipeline phân loại ảnh crop thành 6 lớp với ResNet50 hoặc EfficientNet-B0. Dữ liệu lấy theo split trong `data/train_test_split/labels_split_unk3000.csv`; ảnh trong `data/dataset_6class_crops_v2/` được resize giữ tỷ lệ, padding thành `224×224` và chuẩn hóa theo ImageNet. Không dùng data augmentation ở bất kỳ thiết lập nào.

## Môi trường và chạy

Cài thư viện dự án nếu cần:

```bash
python -m pip install -r requirements.txt
```

Chạy một cấu hình train riêng (ví dụ ResNet50 TH1):

```bash
python deep_learning/train.py --backbone resnet50 --setup th1 --device mps
```

`--device` nhận `mps`, `cpu` hoặc `auto`. Trọng số khởi tạo mặc định là ImageNet pretrained; thêm `--no-pretrained` nếu muốn khởi tạo ngẫu nhiên.

## Hai thiết lập

- **TH1:** backbone đóng băng, chỉ classifier được cập nhật.
- **TH2:** fine-tune toàn bộ backbone và classifier ngay từ epoch đầu, bắt đầu từ trọng số khởi tạo đã chọn (ImageNet pretrained theo mặc định).

Cả hai dùng cùng dữ liệu và preprocessing, không augmentation. Learning rate classifier mặc định `1e-3`; learning rate backbone mặc định `1e-5`.

## Random search

Random search xếp hạng theo validation macro-F1, rồi val loss khi hòa. Mỗi TH/backbone chạy mặc định tối đa 5 cấu hình, và không thể tăng giới hạn đó qua CLI. Không gian thử được rút gọn:

- Classifier learning rate: `3e-4`, `1e-3`, `3e-3`.
- Backbone learning rate cho TH2: `1e-5`, `3e-5`.
- Weight decay: `1e-4`, `1e-3`.
- Batch size: `16`, `32`.

Chạy random search riêng cho từng tổ hợp (mỗi lệnh thử tối đa 5 cấu hình):

```bash
python deep_learning/random_search.py --setups th1 --backbones resnet50 --device mps
python deep_learning/random_search.py --setups th2 --backbones resnet50 --device mps
python deep_learning/random_search.py --setups th1 --backbones efficientnet_b0 --device mps
python deep_learning/random_search.py --setups th2 --backbones efficientnet_b0 --device mps
```

Có thể giảm số lượt cho mỗi lệnh, ví dụ:

```bash
python deep_learning/random_search.py --setups th2 --backbones resnet50 --configs-per-backbone 3 --device mps
```

`--configs-per-backbone` nhận giá trị 1–5. Search dùng seed lấy mẫu `42` và seed huấn luyện `42`. Sau khi chọn cấu hình theo validation, script khóa run thắng và đánh giá test một lần. Không dùng test để chọn hyperparameter.

## Kết quả

Mỗi run được lưu trong `deep_learning/outputs/`, gồm `config.json`, `training.txt`, `history.csv`, `metrics.json`, `report.txt` và `best.pt`. `results_summary.txt` tổng hợp các run; random search tạo thêm `random_search_summary.txt`. Run cũ không bị ghi đè.

Có thể đánh giá test run riêng sau khi khóa:

```bash
python deep_learning/evaluate.py --run deep_learning/outputs/resnet50_th1_seed42 --lock-run
python deep_learning/evaluate.py --run deep_learning/outputs/resnet50_th1_seed42 --device mps
```

Dự đoán ảnh bằng checkpoint:

```bash
python deep_learning/predict.py \
  --model deep_learning/outputs/resnet50_th1_seed42/best.pt \
  --image data/dataset_6class_crops_v2/hold/example.jpg \
  --device mps
```

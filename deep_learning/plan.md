# Kế hoạch thí nghiệm Deep Learning

## Mục tiêu

Phân loại ảnh crop người–vật thành sáu lớp bằng ResNet50 và EfficientNet-B0. Dùng split cố định trong `data/train_test_split/labels_split_unk3000.csv`; train để cập nhật trọng số, validation để early stopping/chọn cấu hình, test chỉ đánh giá sau khi khóa lựa chọn.

## Thiết lập

| TH | Backbone | Cập nhật trọng số | Khởi tạo | Augmentation |
|---|---|---|---|---|
| TH1 | ResNet50 / EfficientNet-B0 | Chỉ classifier; backbone đóng băng | ImageNet pretrained mặc định | Không |
| TH2 | ResNet50 / EfficientNet-B0 | Toàn bộ model từ epoch đầu | ImageNet pretrained mặc định | Không |

Cả hai dùng resize giữ tỷ lệ, padding `224×224`, rồi chuẩn hóa ImageNet. Có thể dùng `--no-pretrained` để khởi tạo ngẫu nhiên, nhưng mặc định dùng trọng số ImageNet.

## Huấn luyện và chọn model

- Loss là weighted cross entropy; class weight chỉ tính từ train.
- Dùng AdamW, scheduler theo validation loss, early stopping theo validation macro-F1.
- Lưu checkpoint tốt nhất theo validation macro-F1; nếu hòa, ưu tiên validation loss thấp hơn.
- Không dùng test khi huấn luyện, early stopping hoặc tìm hyperparameter.
- `random_search.py` lấy tối đa 5 cấu hình mỗi TH/backbone; miền thử gồm classifier LR `(3e-4, 1e-3, 3e-3)`, backbone LR cho TH2 `(1e-5, 3e-5)`, weight decay `(1e-4, 1e-3)`, batch size `(16, 32)`.
- Sau khi chọn cấu hình thắng trên validation, đánh giá test đúng một lần.

## Các file

- `data.py`: đọc/kiểm tra CSV, Dataset và preprocessing cố định.
- `model.py`: ResNet50, EfficientNet-B0 và điều khiển freeze backbone.
- `config.py`: cấu hình, seed, device và tổng hợp kết quả.
- `train.py`: train/validation, checkpoint và early stopping.
- `random_search.py`: lấy mẫu cấu hình và chọn cấu hình theo validation.
- `evaluate.py`: khóa run và đánh giá test.
- `predict.py`: dự đoán ảnh từ checkpoint.

# Phân loại ảnh tương tác người–vật

Pipeline này phân loại **ảnh đã crop sẵn** thành 6 lớp bằng ResNet50 hoặc EfficientNet-B0. Code đọc ảnh trong `data/dataset_6class_crops_v2/` theo split ở `data/train_test_split/labels_split_unk3000.csv`.

Ảnh nguồn không bị crop lại, ghi đè hay lưu bản biến đổi. Khi nạp ảnh, pipeline chỉ resize giữ nguyên tỷ lệ, padding thành `224×224` và chuẩn hóa theo ImageNet. TH2/TH3 bổ sung lật ngang và ColorJitter nhẹ.

## Môi trường

Yêu cầu Python với PyTorch và torchvision tương thích. Cài các thư viện dự án nếu môi trường chưa có:

```bash
python -m pip install -r requirements.txt
```

Trên MacBook Air M2, dùng Python native `arm64`, bản PyTorch macOS có MPS và **không cài CUDA**. Kiểm tra MPS:

```bash
python -c "import platform, torch, torchvision; print(platform.machine(), torch.__version__, torchvision.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())"
```

Trọng số ImageNet pretrained được tải khi tạo model lần đầu, nên cần kết nối mạng lúc đó.

## Chạy baseline

Lệnh dưới đây chạy sáu cấu hình (TH1–TH3 × hai backbone), batch size 16 trên MPS:

```bash
python deep_learning/train.py --all-baselines --device mps --batch-size 16
```

Hoặc chạy riêng một cấu hình, ví dụ ResNet50 TH1:

```bash
python deep_learning/train.py \
  --backbone resnet50 \
  --setup th1 \
  --seed 42 \
  --device mps \
  --batch-size 16
```

`--device` nhận `mps`, `cpu` hoặc `auto`. Chọn `mps` trên M2 để chương trình dừng kèm thông báo nếu MPS không khả dụng; `auto` tự chọn MPS nếu có, nếu không sẽ dùng CPU. Mặc định DataLoader dùng `num_workers=0`.

## Huấn luyện theo ba thiết lập

Cả ba thiết lập đều khởi tạo backbone bằng trọng số ImageNet và thay classifier cuối bằng 6 đầu ra. `--backbone` chọn `resnet50` hoặc `efficientnet_b0`; các lệnh bên dưới minh họa ResNet50. Đổi giá trị này để chạy cùng thiết lập trên EfficientNet-B0.

### TH1 — Backbone đóng băng, không augmentation

Chỉ classifier được cập nhật. Dùng làm mốc transfer learning với đặc trưng backbone cố định. Learning rate classifier mặc định `1e-3`:

```bash
python deep_learning/train.py \
  --backbone resnet50 \
  --setup th1 \
  --seed 42 \
  --device mps \
  --batch-size 16 \
  --epochs 30 \
  --patience 6 \
  --classifier-lr 1e-3 \
  --weight-decay 1e-4
```

### TH2 — Backbone đóng băng, augmentation nhẹ

Chỉ classifier được cập nhật như TH1. Train bổ sung lật ngang ngẫu nhiên và ColorJitter nhẹ; val không augmentation. Thiết lập này đo ảnh hưởng augmentation khi backbone giữ cố định:

```bash
python deep_learning/train.py \
  --backbone resnet50 \
  --setup th2 \
  --seed 42 \
  --device mps \
  --batch-size 16 \
  --epochs 30 \
  --patience 6 \
  --classifier-lr 1e-3 \
  --weight-decay 1e-4
```

### TH3 — Fine-tune toàn bộ backbone

TH3 dùng cùng augmentation nhẹ như TH2. Backbone đóng băng trong hai epoch đầu để classifier ổn định, sau đó mở toàn bộ backbone ở epoch 3. Mặc định learning rate classifier là `1e-4`, backbone là `1e-5`:

```bash
python deep_learning/train.py \
  --backbone resnet50 \
  --setup th3 \
  --seed 42 \
  --device mps \
  --batch-size 16 \
  --epochs 30 \
  --patience 6 \
  --unfreeze-epoch 3 \
  --classifier-lr 1e-4 \
  --backbone-lr 1e-5 \
  --weight-decay 1e-4
```

Có thể thay learning rate, batch size hoặc `--unfreeze-epoch` qua CLI khi dò tham số trên train/val. `train.py` không nạp split test. Với mỗi thiết lập, chương trình tạo run riêng và không ghi đè kết quả cũ.

## Dò hyperparameter và chọn cấu hình

Chạy sáu baseline trước để có mốc so sánh. Sau đó thử từng nhóm tham số trên cùng split và seed; mỗi lệnh tạo một run mới. Chỉ dùng kết quả **validation** để điều chỉnh và chọn tham số. Code hiện chạy từng cấu hình bằng CLI, chưa tự sinh lưới tham số.

| Tham số | Giá trị baseline | Giá trị nên thử | Áp dụng |
|---|---:|---|---|
| Learning rate classifier | TH1/TH2: `1e-3`; TH3: `1e-4` | TH1/TH2: `3e-4`, `1e-3`, `3e-3`; TH3: `1e-4`, `3e-4`, `1e-3` | `--classifier-lr` |
| Learning rate backbone | `1e-5` | `1e-5`, `3e-5`, `1e-4` | TH3: `--backbone-lr` |
| Weight decay | `1e-4` | `1e-5`, `1e-4`, `1e-3` | `--weight-decay` |
| Batch size | `16` | `16`, `32`; giảm còn `8` hoặc `4` nếu thiếu bộ nhớ | `--batch-size` |
| Epoch tối đa / patience | `30` / `6` | Giữ baseline trước; chỉ đổi nếu history cho thấy dừng quá sớm hoặc còn cải thiện | `--epochs`, `--patience` |
| Epoch mở backbone | TH3: `3` | Cố định ở epoch `3`; backbone đóng băng trong hai epoch đầu | TH3: `--unfreeze-epoch` |

Các lệnh ví dụ để so sánh learning rate classifier của ResNet50 TH1 (giữ các tham số khác như nhau):

```bash
python deep_learning/train.py --backbone resnet50 --setup th1 --seed 42 --device mps --classifier-lr 3e-4
python deep_learning/train.py --backbone resnet50 --setup th1 --seed 42 --device mps --classifier-lr 1e-3
python deep_learning/train.py --backbone resnet50 --setup th1 --seed 42 --device mps --classifier-lr 3e-3
```

Với TH3, dò riêng learning rate classifier và backbone; mỗi lượt chỉ thay một tham số. Sau khi chọn learning rate, lần lượt thử weight decay rồi batch size. Chỉ thử thay đổi augmentation khi có câu hỏi cụ thể: code hiện hỗ trợ sẵn flip ngang và ColorJitter cho TH2/TH3, chưa có CLI cho rotation hoặc tắt riêng từng phép biến đổi. Dropout và class weight dạng lũy thừa cũng chưa được triển khai để dò.

Chọn run có **val macro-F1 cao nhất**; nếu điểm gần bằng nhau, xem val loss, balanced accuracy và F1 từng lớp để tránh chọn cấu hình làm giảm mạnh một lớp. `results_summary.txt` tự đánh dấu run tốt nhất hiện tại cho từng backbone/TH, còn `history.csv` và `metrics.json` lưu chi tiết từng run. Chỉ khóa run sau khi đã chọn xong; không dùng test để điều chỉnh tham số.

Random search giới hạn tối đa 10 cấu hình cho **mỗi cặp TH × backbone**. Chạy riêng từng cặp theo thứ tự dưới đây:

```bash
python deep_learning/random_search.py --setups th1 --backbones resnet50 --device mps
python deep_learning/random_search.py --setups th1 --backbones efficientnet_b0 --device mps
python deep_learning/random_search.py --setups th2 --backbones resnet50 --device mps
python deep_learning/random_search.py --setups th2 --backbones efficientnet_b0 --device mps
python deep_learning/random_search.py --setups th3 --backbones resnet50 --device mps
python deep_learning/random_search.py --setups th3 --backbones efficientnet_b0 --device mps
```

Mỗi lệnh trên chọn tối đa 10 cấu hình ngẫu nhiên, nên mỗi TH có tối đa 20 run (10 cho mỗi backbone), và cả sáu cặp có tối đa 60 run. Mỗi cấu hình chỉ chạy một lần với seed 42. TH3 chỉ dùng `unfreeze_epoch=3`.

Có thể tăng tốc bằng cách giảm số cấu hình, ví dụ `--configs-per-backbone 6` (giá trị hợp lệ 1–10). `--search-seed` điều khiển việc lấy mẫu ngẫu nhiên, còn seed huấn luyện vẫn là 42.

Sau khi xếp hạng xong bằng validation, script tự khóa run thắng và đánh giá test đúng một lần cho mỗi cặp TH/backbone được chạy. Test không tham gia chọn hyperparameter. Kết quả train/val/test nằm trong `metrics.json` và `report.txt` của run thắng. Script cũng tạo `deep_learning/outputs/random_search_summary.txt` (nếu đã tồn tại thì tạo tên có hậu tố), ghi cấu hình thắng, các kết quả và đường dẫn checkpoint; toàn bộ run được liệt kê trong `results_summary.txt`. Vì random search đã tự đánh giá test, không cần chạy `evaluate.py` cho các run thắng này.

## Kết quả và đánh giá test

Mỗi lần chạy tạo thư mục duy nhất trong `deep_learning/outputs/`, gồm cấu hình, manifest mẫu, `training.txt`, lịch sử epoch, metrics/report train–val và `best.pt` được chọn theo val macro-F1. `training.txt` ghi cấu hình, runtime, từng epoch, metric và checkpoint tốt nhất; file được cập nhật ngay sau mỗi sự kiện để có thể mở theo dõi trong lúc chạy. Khi đánh giá test, file cũng ghi kết quả test. `results_summary.txt` cập nhật sau mỗi epoch; `random_search_summary.txt` cập nhật sau mỗi cấu hình hoàn tất và sau mỗi lượt đánh giá test. Run cũ không bị ghi đè. Thư mục outputs/checkpoint đã được thêm vào `.gitignore`.

File `deep_learning/outputs/results_summary.txt` được tạo/cập nhật tự động sau mỗi lần train, khóa run hoặc đánh giá test. File này liệt kê tham số, epoch tốt nhất, kết quả train/val/test và đường dẫn checkpoint của từng run; đồng thời đánh dấu run có val macro-F1 cao nhất cho mỗi tổ hợp backbone/TH (nếu hòa, chọn val loss thấp hơn). Test chỉ hiện sau khi run đó đã được đánh giá.

Nếu huấn luyện một run riêng bằng `train.py` (không qua random search), sau khi xem train/val có thể khóa run:

```bash
python deep_learning/evaluate.py --run deep_learning/outputs/resnet50_th3_seed42 --lock-run
```

Rồi đánh giá test đúng một lần:

```bash
python deep_learning/evaluate.py --run deep_learning/outputs/resnet50_th3_seed42 --device mps
```

Lệnh trên thêm kết quả test vào `metrics.json` và `report.txt`. Các run thắng do `random_search.py` chọn đã được khóa và đánh giá tự động. Không dùng kết quả test để thay đổi lựa chọn hyperparameter.

## Dự đoán ảnh

Một ảnh:

```bash
python deep_learning/predict.py \
  --model deep_learning/outputs/resnet50_th3_seed42/best.pt \
  --image data/dataset_6class_crops_v2/hold/example.jpg \
  --device mps
```

Nhiều ảnh có thể truyền sau một `--image`; mỗi ảnh được xuất thành một dòng JSON. Confidence là xác suất softmax thô, chưa hiệu chuẩn. Dự đoán dùng đúng resize-padding và chuẩn hóa như validation.

## Tham số chính

| Tham số | Mặc định | Ý nghĩa |
|---|---:|---|
| `--backbone` | `resnet50` | `resnet50` hoặc `efficientnet_b0` |
| `--setup` | `th1` | `th1`, `th2` hoặc `th3` |
| `--seed` | `42` | Seed tái lập |
| `--batch-size` | `16` | Batch size |
| `--epochs` | `30` | Số epoch tối đa |
| `--patience` | `6` | Early stopping theo val macro-F1 |
| `--unfreeze-epoch` | `3` | Epoch mở backbone ở TH3 |
| `--classifier-lr` | theo TH | Learning rate classifier |
| `--backbone-lr` | `1e-5` | Learning rate backbone ở TH3 |
| `--weight-decay` | `1e-4` | Weight decay AdamW |
| `--device` | `auto` | `mps`, `cpu` hoặc `auto` |


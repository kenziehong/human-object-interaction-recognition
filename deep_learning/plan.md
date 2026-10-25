# Kế hoạch triển khai Deep Learning

## 1. Mục tiêu và phạm vi

Xây dựng pipeline phân loại ảnh crop người thành 6 lớp bằng PyTorch, so sánh ba thiết lập thực nghiệm (TH1–TH3) trên hai backbone pretrained ImageNet:

- ResNet50
- EfficientNet-B0

Tổng cộng có 6 cấu hình chính. Dùng đúng các split đã có trong `data/train_test_split/labels_split_unk3000.csv`; ảnh được đọc từ `data/dataset_6class_crops_v2/`. `crop_path` trong CSV là đường dẫn tương đối, ví dụ `unknown/<tên_ảnh>.jpg`; nhãn `eat/drink` tương ứng thư mục `eat_drink`.

**Quy ước dữ liệu:** chỉ `train` được dùng để cập nhật trọng số. `val` dùng để theo dõi, early stopping và chọn cấu hình/checkpoint. `test` không tham gia huấn luyện, early stopping hay dò tham số; chỉ chạy đánh giá cuối sau khi đã khóa cấu hình. Không gộp val/test vào train trong các thí nghiệm chính.

## 2. Thiết kế thực nghiệm

| TH | Backbone | Augmentation train | Backbone được cập nhật | Mục đích |
|---|---|---|---|---|
| TH1 | ResNet50 / EfficientNet-B0 | Không | Đóng băng; chỉ học classifier | Mốc transfer learning với đặc trưng cố định |
| TH2 | ResNet50 / EfficientNet-B0 | Nhẹ | Đóng băng; chỉ học classifier | Đo tác động augmentation khi backbone cố định |
| TH3 | ResNet50 / EfficientNet-B0 | Nhẹ | Fine-tune toàn bộ backbone và classifier | Đo lợi ích thích nghi backbone trên bài toán |

Khởi tạo mỗi backbone bằng trọng số ImageNet, thay classifier cuối thành một lớp `Linear` có 6 đầu ra. ResNet50 dùng đặc trưng 2.048 chiều; EfficientNet-B0 dùng 1.280 chiều. Giữ cùng quy trình tiền xử lý, loss, cách chọn checkpoint và metric cho cả hai backbone để phép so sánh nhất quán.

## 3. Cấu trúc thư mục dự kiến

```text
deep_learning/
├── plan.md                 # Kế hoạch này
├── config.py               # Đường dẫn, class order, device, seed, hyperparameter
├── data.py                 # CSV, Dataset, transforms, kiểm tra split/ảnh
├── model.py                # ResNet50/EfficientNet-B0, classifier, freeze/unfreeze
├── train.py                # Train, validation, early stopping, dò cấu hình
├── random_search.py        # Lấy mẫu hyperparameter và chọn cấu hình theo val macro-F1
├── evaluate.py             # Metrics/report; test chỉ chạy sau khi khóa cấu hình
├── predict.py              # Nạp checkpoint và dự đoán ảnh theo JSON
└── outputs/                 # Sinh tự động; không commit checkpoint lớn
    └── <model_name>/        # Ví dụ resnet50_th1_seed42
        ├── config.json       # Cấu hình và metadata của run
        ├── training.txt      # Tiến trình runtime, epoch và checkpoint theo thời gian
        ├── history.csv       # Metrics tổng thể theo epoch (train/val)
        ├── metrics.json      # Metrics tổng thể và từng lớp cho train/val/test
        ├── report.txt        # Classification reports và confusion matrices
        └── best.pt           # Checkpoint tốt nhất theo val macro-F1, dùng lại để predict
    └── results_summary.txt   # Tổng hợp tham số, metrics và checkpoint của các run
```

Mỗi run có tên duy nhất theo backbone, TH, seed và/hoặc mã cấu hình; không ghi đè run cũ. Chạy sáu cấu hình chính sẽ giữ checkpoint tốt nhất riêng (ResNet50/EfficientNet-B0 × TH1/TH2/TH3), mỗi checkpoint được chọn bằng val macro-F1. `config.json` lưu metadata; `training.txt` theo dõi tiến trình; `history.csv` theo dõi metrics mỗi epoch; `metrics.json` lưu số liệu máy đọc được; `report.txt` để xem per-class report/confusion matrix; `best.pt` là checkpoint dùng cho evaluate và predict. Trước khi đánh giá test, `metrics.json` và `report.txt` chỉ có train/val; sau đó kết quả test được bổ sung vào cùng hai file. Đưa `outputs/`, checkpoint và cache vào `.gitignore`.

## 4. Các bước triển khai

### Bước 1 — Kiểm tra dữ liệu và split

`data.py` đọc `labels_split_unk3000.csv`, ánh xạ `label` theo thứ tự cố định `hold`, `ride`, `eat/drink`, `talk_on_phone`, `work_on_computer`, `unknown`, rồi ghép `crop_path` với `data/dataset_6class_crops_v2/`. Không suy ra split từ tên thư mục; dùng duy nhất cột `split`.

Trước khi train, in và lưu:

- Số dòng và số ảnh tồn tại theo split, lớp và nguồn (`dataset`).
- Số ảnh thiếu, nhãn không hợp lệ, split không hợp lệ, đường dẫn trùng.
- Kiểm tra mỗi `source_image` chỉ xuất hiện trong một split (nếu cột có trong CSV).
- Tỷ lệ lớp và xác nhận đủ sáu lớp trong train/val/test.

Dừng với thông báo rõ ràng nếu có ảnh thiếu/không đọc được, không âm thầm bỏ qua như pipeline feature cũ. Lưu manifest của các mẫu thực sự dùng để có thể tái lập. Giữ nguyên phép chia trong CSV, kể cả khi số lượng thực tế khác thống kê lịch sử.

### Bước 2 — Dataset và tiền xử lý ảnh

- Dùng `torch.utils.data.Dataset` đọc CSV; không dùng `ImageFolder` để tránh mất metadata và phụ thuộc thứ tự thư mục.
- Đọc RGB, xử lý ảnh lỗi thành lỗi có thông tin đường dẫn.
- Ảnh trong `data/dataset_6class_crops_v2/` đã được crop theo người; dùng nguyên crop đó làm đầu vào, **không crop lại và không ghi đè/tạo lại ảnh**. Chỉ resize giữ tỷ lệ và padding trong transform lúc nạp ảnh để tạo tensor `224×224`; không lưu ảnh đã biến đổi ra đĩa và không dùng center/random crop làm mất nội dung.
- Chuẩn hóa theo ImageNet: mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`.
- TH1: không augmentation ngẫu nhiên.
- TH2/TH3 dùng cùng một cấu hình augmentation để phép so sánh TH2–TH3 chỉ khác trạng thái backbone:
  - Horizontal flip `p=0.5`. Nhãn 6 lớp không phụ thuộc trái/phải; phép lật giữ tương quan người–vật.
  - Color jitter nhẹ: brightness `0.1`, contrast `0.1`, saturation `0.1`, hue `0.02`; mục đích là bớt phụ thuộc ánh sáng/nguồn ảnh mà vẫn giữ màu và đặc điểm vật thể tương đối.
  - Rotation không bật trong baseline. Nếu xem mẫu train thấy góc nghiêng đáng kể, làm một ablation riêng với rotation ±5°; chỉ giữ phép này nếu val macro-F1 không giảm và per-class F1 ổn định. Không cần thử ±10° ngay.
- Val/test chỉ có resize-padding và normalization cố định, không augmentation ngẫu nhiên.

Vì nhãn là tương tác người–vật, cần giữ bàn tay, vật và vị trí tương đối giữa chúng. Không dùng `RandomResizedCrop`, `RandomCrop`, `CenterCrop`, vertical flip, perspective/elastic distortion, rotation lớn, `RandomErasing`, MixUp hoặc CutMix trong cấu hình khởi đầu: chúng có thể cắt/méo vật nhỏ hoặc tạo/che mất quan hệ tương tác. Torchvision định nghĩa `RandomResizedCrop` là lấy một vùng ngẫu nhiên rồi resize, nên không phù hợp với crop người khi vùng quyết định nhãn có thể nhỏ ([Torchvision RandomResizedCrop](https://docs.pytorch.org/vision/main/generated/torchvision.transforms.v2.RandomResizedCrop.html)). Nghiên cứu HOI cũng xác định bố cục không gian người–vật và ngữ cảnh là cue quan trọng; vì vậy ưu tiên các biến đổi bảo toàn hình học ([HO-RCNN](https://arxiv.org/abs/1702.05448), [ContextHOI](https://ojs.aaai.org/index.php/AAAI/article/view/32411)).

### Bước 2a — Chạy trên GPU MacBook Air M2

PyTorch dùng backend Metal Performance Shaders (`mps`) để chạy model/tensor trên GPU Apple Silicon. Triển khai `--device mps|cpu|auto`, nhưng trên máy M2 lệnh chạy thực nghiệm phải chọn `--device mps`; nếu MPS không khả dụng thì dừng sớm với hướng dẫn kiểm tra môi trường, không âm thầm tiếp tục trên CPU. PyTorch cung cấp `torch.backends.mps.is_available()` để kiểm tra và yêu cầu đưa cả model lẫn batch tensor lên cùng `torch.device("mps")` ([tài liệu MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)).

- Dùng Python native `arm64` và bản PyTorch/torchvision tương thích macOS; cài theo hướng dẫn chính thức của PyTorch. Không cài bản CUDA cho M2.
- Khi khởi động, ghi `torch.__version__`, `torchvision.__version__`, `platform.machine()`, `torch.backends.mps.is_built()`, `torch.backends.mps.is_available()` và device thực tế của model/input vào log; chạy một forward/backward batch smoke test trước baseline.
- Bắt đầu batch size `16`; thử `32` nếu MPS memory ổn định. Nếu thiếu bộ nhớ, giảm về `8` rồi `4`.
- Dùng `DataLoader(num_workers=0)` lúc đầu trên macOS, sau đó chỉ tăng lên `2` nếu việc đọc ảnh là nút thắt; `pin_memory=False` cho MPS. Tránh copy tensor CPU↔GPU lặp lại trong từng layer/metric.
- Giữ FP32 làm mặc định để ưu tiên tương thích. Chỉ bật autocast/mixed precision nếu đã xác nhận trên phiên bản PyTorch đang cài rằng MPS hỗ trợ các phép toán của model, loss và backward, và kết quả/độ ổn định không suy giảm.
- Đồng bộ MPS trước khi đo thời gian epoch để timing phản ánh thời gian thực thi kernel; log thời gian epoch, ảnh/giây và batch size.
- Nếu gặp phép toán chưa hỗ trợ, ghi rõ fallback và đo lại; không bật CPU fallback âm thầm vì có thể làm chậm đáng kể. Có thể chạy `--device cpu` một cách tường minh để debug.

PyTorch xác nhận backend `mps` dùng Metal/MPS cho phép chạy model và tensor trên GPU; kiểm tra build/khả dụng bằng API backend ([MPS backend](https://docs.pytorch.org/docs/stable/notes/mps.html), [API kiểm tra backend](https://docs.pytorch.org/docs/stable/backends.html)).

Lệnh chạy trên Mac M2 sau khi các script được viết:

```bash
python -c "import platform, torch; print(platform.machine(), torch.__version__, torch.backends.mps.is_built(), torch.backends.mps.is_available())"
python deep_learning/train.py --all-baselines --device mps --batch-size 16
```

### Bước 3 — Model và chiến lược tối ưu

- Tạo backbone từ `torchvision` pretrained ImageNet; thay riêng classifier bằng `Linear(in_features, 6)`.
- TH1/TH2: `requires_grad=False` cho backbone; chỉ classifier vào optimizer.
- TH3: khởi động 2–3 epoch đầu với backbone đóng băng để classifier ổn định, sau đó mở toàn bộ backbone với learning rate riêng nhỏ hơn cho backbone. Lưu rõ epoch mở khóa trong config.
- Loss: `CrossEntropyLoss` với class weight nghịch tần suất tính **chỉ trên train**; bắt đầu không dùng đồng thời `WeightedRandomSampler` để tránh cân bằng quá mức.
- Optimizer khởi đầu: AdamW, weight decay `1e-4`.
- Batch size thử `16`, tăng `32` nếu bộ nhớ cho phép. Mặc định dùng FP32; mixed precision chỉ bật sau khi kiểm chứng riêng trên MPS theo mục 2a.

### Bước 4 — Training loop, validation và early stopping

Mỗi epoch: train trên train, sau đó dự đoán toàn bộ train và val trong `eval()`/`no_grad()` để tính report nhất quán với kết quả hiện có. Ghi train loss, val loss và bộ metrics dưới đây; chọn checkpoint chỉ theo val macro-F1.

Metrics phải tương thích với định dạng `embedding_ml/outputs/all_results_v2.txt`, được xuất cho **train, val và test** (test chỉ sau khi khóa thí nghiệm):

- Tổng thể: `accuracy`, `precision_macro`, `recall_macro`, `f1_macro`, `precision_weighted`, `recall_weighted`, `f1_weighted`.
- Từng lớp trong thứ tự cố định: `precision`, `recall`, `f1-score`, `support`.
- Báo cáo văn bản kiểu `classification_report`, với các dòng từng lớp, `macro avg`, `weighted avg`, tổng support và accuracy.
- Lưu thêm confusion matrix (số lượng tuyệt đối, hàng = nhãn thật, cột = dự đoán), val loss, balanced accuracy và macro-F1 search score. Các metric bổ sung không thay thế bộ metric trên.

`history.csv` có một dòng mỗi epoch, gồm loss, accuracy, macro/weighted precision-recall-F1 của train/val, val balanced accuracy, learning rate, thời gian và ảnh/giây. Không cần lưu per-class report ở mọi epoch; sau khi chọn checkpoint, ghi đầy đủ per-class precision/recall/F1/support, macro/weighted averages và confusion matrix cho train/val/test trong `metrics.json` và `report.txt`. Trong mọi report ghi class order và support để tránh nhầm thứ tự nhãn.

- Metric chính để chọn checkpoint: **val macro-F1**, do lớp `unknown` chiếm tỷ lệ lớn và accuracy có thể che khuất hiệu quả lớp nhỏ.
- Metric phụ: val loss, accuracy, balanced accuracy, per-class F1 và confusion matrix.
- Lưu `best.pt` khi val macro-F1 cải thiện; checkpoint này dùng để khôi phục run và đánh giá sau train.
- Early stopping khởi điểm: tối đa 30 epoch, patience 5–7 epoch, `min_delta` cấu hình được. Scheduler `ReduceLROnPlateau` theo val loss hoặc val macro-F1; chọn một quy ước thống nhất và lưu lại.
- Xử lý hòa điểm bằng val loss thấp hơn; không chọn checkpoint bằng bất kỳ metric test nào.

### Bước 5 — Theo dõi để điều chỉnh mô hình

Ghi metrics theo epoch vào `history.csv`, kết quả cuối vào `metrics.json`. Có thể thêm TensorBoard khi cần xem đường học tương tác, nhưng CSV vẫn là log chính. Khi so sánh nhiều run, tổng hợp các dòng `metrics.json` thành một bảng `experiments.csv` dùng chung (không tạo thư mục/file phụ cho từng run). Bảng gồm TH, backbone, seed, hyperparameter, epoch tốt nhất, metric val, thời gian, device (MPS/CPU), ảnh/giây và đường dẫn run/checkpoint.

Theo dõi các dấu hiệu sau để quyết định lượt thử tiếp:

- Train loss giảm nhưng val loss tăng/val macro-F1 giảm: overfit; tăng regularization/augmentation nhẹ, giảm số epoch hoặc giảm learning rate.
- Train và val đều kém, gần nhau: underfit; thử learning rate phù hợp hơn, mở backbone (TH3), tăng số epoch hoặc giảm regularization.
- Macro-F1 ổn nhưng recall lớp nhỏ thấp: xem confusion matrix và từng lớp; điều chỉnh class weight có giới hạn, không chỉ tối ưu accuracy.
- GPU utilization thấp/ảnh mỗi giây thấp: kiểm tra log device/model/input, đo riêng thời gian DataLoader và forward/backward; tăng workers tối đa từng bước hoặc batch size nếu memory cho phép.
- Val metric dao động mạnh: cố định seed, dùng patience phù hợp, chạy lại cấu hình ứng viên với 3 seed trước khi kết luận.
- Ghi lại mọi lượt chạy, kể cả lượt xấu; không sửa kết quả sau khi xem test.

## 5. Kế hoạch dò hyperparameter

Không chạy toàn bộ tích Descartes của các miền tham số. `random_search.py` lấy mẫu tối đa 10 cấu hình cho mỗi cặp TH/backbone, dùng seed 42 và xếp hạng chỉ trên validation. Nếu cần báo cáo độ ổn định qua nhiều seed, chạy lại cấu hình ứng viên riêng sau bước tìm kiếm.

| Tham số | Giá trị khởi đầu | Các giá trị/miền thử có kiểm soát |
|---|---:|---|
| Batch size | 16 | 16, 32 (theo bộ nhớ) |
| LR classifier, TH1/TH2 | `1e-3` | `3e-4`, `1e-3`, `3e-3` |
| LR backbone, TH3 | `1e-5` | `1e-5`, `3e-5`, `1e-4` |
| LR classifier, TH3 | `1e-4` | `1e-4`, `3e-4`, `1e-3` |
| Weight decay | `1e-4` | `1e-5`, `1e-4`, `1e-3` |
| Dropout classifier (nếu thêm) | `0` | so sánh `0` với `0.2`; không thêm tầng Dense mới ở baseline |
| Augmentation TH2/TH3 | flip `p=.5` + ColorJitter `(0.1, 0.1, 0.1, 0.02)` | ablation riêng rotation ±5° hoặc giảm/bỏ ColorJitter; chỉ thay một yếu tố mỗi lượt |
| Class weight | nghịch tần suất train | baseline; thử làm mềm trọng số bằng lũy thừa `1/2` nếu recall/F1 lớp nhỏ xấu |

Trình tự dò đề xuất:

1. Tìm LR (classifier và backbone tách riêng ở TH3) trên baseline augmentation/loss.
2. Giữ LR tốt, so sánh weight decay và class weight.
3. So sánh baseline flip+ColorJitter với một biến thể mỗi lần (chỉ rotation ±5°, hoặc giảm/bỏ ColorJitter); không chạy các phép crop mạnh.
4. Kiểm tra batch size và scheduler/patience nếu đường học cho thấy cần thiết.
5. Chọn cấu hình tốt theo val macro-F1; kiểm tra độ ổn định qua seed bổ sung là bước riêng, không thuộc giới hạn 10 cấu hình của random search.

Với mỗi nhóm so sánh, cố định các yếu tố khác để hiểu nguyên nhân thay đổi. Tuning chỉ dùng train/val. Ghi trước tiêu chí chọn: val macro-F1 cao nhất; nếu gần tương đương thì ưu tiên val loss thấp hơn và độ ổn định qua seed tốt hơn. Nếu cần thêm lượt sau khi xem test thì test đã trở thành dữ liệu phát triển; phải tạo một test mới độc lập để có đánh giá cuối không thiên lệch.

## 6. Khóa mô hình và đánh giá test

Sau khi hoàn tất dò tham số cho một nhóm TH/backbone, khóa checkpoint thắng theo val. `random_search.py` thực hiện bước này và đánh giá checkpoint thắng trên test đúng một lượt, không cập nhật trọng số. Báo cáo:

- Test macro-F1, accuracy, balanced accuracy, macro precision/recall.
- Precision, recall, F1 và support của từng lớp.
- Confusion matrix có tên lớp cố định.
- Backbone, TH, checkpoint/epoch, seed, hyperparameter và tỷ lệ lớp test.
- Toàn bộ metrics train/val/test cùng per-class report và support, để thấy khoảng cách train–val–test và đối chiếu trực tiếp với `all_results_v2.txt`.
- So sánh cả sáu cấu hình theo cách chọn checkpoint bằng val; nếu đánh giá test đủ sáu để làm bảng nghiên cứu, không được dùng thứ hạng test để đổi mô hình thắng cuộc.

Không huấn luyện lại trên train+val trong thiết kế chính, vì yêu cầu dùng val cho early stopping/chọn checkpoint và cần giữ vai trò độc lập của test. Nếu sau này cần model triển khai tối đa dữ liệu, có thể thêm quy trình refit riêng: chốt epoch từ thí nghiệm train/val, train lại trên train+val với epoch cố định, rồi chỉ test một lần; phải ghi rõ đây là một giao thức riêng.

## 7. Thứ tự viết code và tiêu chí hoàn tất

1. `config.py`: đường dẫn, thứ tự lớp, seed, defaults và nhận tham số CLI/config.
2. `data.py`: đọc manifest, kiểm tra split/ảnh, dataset và transforms.
3. `model.py`: hai backbone, classifier 6 lớp, freeze/unfreeze.
4. `train.py`: train/validation, metrics mỗi epoch, scheduler, early stopping; chạy baseline và các lượt dò bằng config CLI.
5. `evaluate.py`: nhận run/checkpoint đã khóa; thêm metrics test vào `metrics.json`, `report.txt` và `training.txt`. `random_search.py` gọi bước này tự động cho cấu hình thắng.
6. `predict.py`: nhận đường dẫn checkpoint và ảnh, nạp kiến trúc/weights/preprocessing/class order từ checkpoint, rồi xuất kết quả theo JSON.
7. Thêm `README.md` chỉ khi cần hướng dẫn chạy riêng; các hướng dẫn cốt lõi có thể để trong đầu `plan.md` khi repo còn nhỏ.

### Giao diện dự đoán

`predict.py` nhận đường dẫn đến một `best.pt` và đường dẫn đến một hay nhiều ảnh. Tệp checkpoint phải chứa state dict, backbone, class order, preprocessing cần thiết và tên mô hình; không cần dò lại cấu hình hay huấn luyện khi dự đoán. Dùng cùng transform resize-padding/normalization như lúc validation. Với mỗi ảnh, lấy lớp có xác suất softmax cao nhất làm dự đoán và xuất `confidence` bằng xác suất đó (đây là confidence thô, chưa hiệu chuẩn).

Ví dụ lệnh:

```bash
python deep_learning/predict.py \
  --model deep_learning/outputs/resnet50_th1_seed42/best.pt \
  --image data/vcoco_6class_crops/test/talk_on_phone/6771_1301774.jpg
```

Một ảnh thì in một JSON object; nhiều ảnh thì in JSON Lines, một object mỗi ảnh. Schema tương thích với pipeline embedding hiện có:

```json
{"image_path":"data/vcoco_6class_crops/test/talk_on_phone/6771_1301774.jpg","label_id":3,"label":"talk_on_phone","confidence":0.9483333333333335,"model_name":"resnet50_th1_seed42"}
```

`model_name` lấy từ metadata checkpoint; `label_id` dựa trên class order cố định đã lưu trong checkpoint. Khi nạp model, `predict.py` tự chọn MPS nếu khả dụng hoặc CPU, đồng thời cho phép `--device mps|cpu|auto`.

Pipeline sẵn sàng khi: cùng config/seed có thể tái lập run; split được lấy đúng từ CSV; lỗi đường dẫn được phát hiện trước khi train; có checkpoint tốt nhất cho từng cấu hình và dùng lại được; `config.json`/`best.pt` đủ metadata để dùng lại run; val điều khiển early stopping và chọn cấu hình; test không tham gia huấn luyện/tuning, chỉ đánh giá checkpoint thắng sau khi khóa; `metrics.json`/`report.txt` có đủ metrics tổng thể/từng lớp như `all_results_v2.txt`; các file `.txt` theo dõi được tiến trình; `predict.py` nhận model+ảnh và trả đúng schema JSON; và log xác nhận forward/backward chạy trên MPS của Mac M2.

## 8. Lưu ý khi diễn giải kết quả

`labels_split_unk3000.csv` giới hạn tổng số mẫu `unknown` ở 3.000 trên cả ba split. Vì vậy phân bố test không phải phân bố tự nhiên của bộ dữ liệu đầy đủ; ghi rõ điều này trong báo cáo và không so trực tiếp metric với một test set có tỷ lệ `unknown` khác. Ảnh được gộp từ bốn nguồn nên nên xem thêm metric theo nguồn như phân tích phụ (không thay thế metric tổng thể), để nhận biết khả năng tổng quát hóa giữa nguồn ảnh.

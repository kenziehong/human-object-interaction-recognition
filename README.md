# Human-Object Interaction Recognition (V-COCO, 6 lớp)

Đồ án môn Thị giác máy tính: nhận diện hành động người–vật. Bài toán tham khảo từ DP-HOI ([Li et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Disentangled_Pre-training_for_Human-Object_Interaction_Detection_CVPR_2024_paper.pdf)). Bài báo làm **phát hiện** HOI (định vị người, vật và hành động); đồ án này chỉ làm nhánh **nhận diện hành động** trên ảnh crop người, và dùng phương pháp khác với bài báo.

Phân loại hành động của 1 người trong ảnh crop thành 6 lớp `hold, ride, eat/drink, talk_on_phone, work_on_computer, unknown`
bằng **đặc trưng thủ công** + **học máy cổ điển**.

- 4 đặc trưng cơ bản: **HOG** (dáng người), **LBP** (kết cấu), **Color Histogram HSV** (màu/bối cảnh), **SIFT + Bag of Visual Words** (chi tiết cục bộ).
- Đặc trưng bổ sung, thiết kế theo đặc điểm dữ liệu (vật nhỏ gần đầu, vật bị cắt khỏi crop, tỉ lệ khung người):

  | Tên | Mô tả | Số chiều |
  |---|---|---|
  | `geom` | hình học bbox người: log tỉ lệ w/h, log w, log h, log diện tích, mức nới thực tế 4 phía (người sát mép ảnh) | 8 |
  | `sift_spm` | SIFT-BoVW + spatial pyramid (1x1 + 2x2), giữ vị trí của "từ" | 2500 |
  | `upper` | HOG (1764) + SIFT-BoVW (500) chỉ trên 40% phía trên crop: đầu, tay gần mặt | 2264 |
  | `gist` | năng lượng Gabor 4 tỉ lệ × 8 hướng trên lưới 4x4: bối cảnh | 512 |
  | `skin` | tỉ lệ da (YCrCb) trên lưới 4x4 + diện tích/trọng tâm/độ trải | 21 |

- Phối hợp: `fused` (4 cơ bản), `combo_local` (sift_spm + sift_upper + color + geom + skin), `fused_all` (tất cả). Danh sách bộ đặc trưng nằm trong `FEATURE_SETS` ở `code/models.py`: thêm tổ hợp mới chỉ cần thêm 1 dòng.
- 4 mô hình: **Logistic Regression**, **SVM (RBF)**, **Random Forest**, **XGBoost**. Cân bằng lớp (vì `unknown` chiếm ~65%) bằng `class_weight`; riêng XGBoost dùng `sample_weight`. SVM và XGBoost giảm chiều bằng PCA 256 khi bộ đặc trưng > 1500 chiều.
- Đánh giá bằng **macro-F1** và **recall từng lớp** (không dùng accuracy). Chọn siêu tham số trên **val**, **test** chỉ dùng để báo cáo.

Mô tả dữ liệu: [`data/vcoco_6class_crops/README.md`](data/vcoco_6class_crops/README.md).

> **Repo không chứa ảnh, đặc trưng hay kết quả** (`data/`, `features/`, `results/` nằm trong `.gitignore`; chỉ có file mô tả dữ liệu). Dữ liệu là ảnh crop người cắt từ [V-COCO](https://github.com/s-gupta/v-coco) (ảnh COCO 2014), cách tạo ghi trong README dữ liệu. Đặt dữ liệu vào `data/vcoco_6class_crops/` theo đúng cấu trúc trong đó (`train/val/test/<lớp>/*.jpg` + `labels.csv`). Các số liệu và phiên bản `fXX` / `rXX` nhắc tới bên dưới là kết quả chạy trên máy của người làm đồ án; biểu đồ và bảng vẫn xem được trong output của `code/04_report.ipynb`.

## Cấu trúc

```
human-object-interaction-recognition/
├── data/vcoco_6class_crops/     dữ liệu crop (train/val/test/<lớp>/*.jpg + labels.csv)
├── code/
│   ├── config.py                đường dẫn, danh sách lớp, tham số đặc trưng
│   ├── data_utils.py            đọc labels.csv, đọc ảnh (bỏ qua ảnh không có trên đĩa)
│   ├── features.py              hog, lbp, color, sift-BoVW (+spm, upper), hog_upper, gist, skin, geom
│   ├── models.py                4 mô hình + lưới siêu tham số + FEATURE_SETS (cách phối)
│   ├── versioning.py            quản lý phiên bản, chặn ghi đè, ghi run_info.json
│   ├── 01_extract_features.py   trích đặc trưng -> features/<fXX>/
│   ├── 02_train_select.py       grid search trên val (12 bộ đặc trưng x 4 mô hình) -> results/<rXX>/val_results.csv
│   ├── 03_final_test.py         train lại trên train+val, đánh giá test -> results/<rXX>/
│   └── 04_report.ipynb          báo cáo chính: hiển thị phiên bản mới nhất (hiện là r01)
├── features/<fXX>/              (tự sinh) đặc trưng .npy, nhãn, codebook SIFT, timing_extract.csv, run_info.json
├── results/<rXX>/               (tự sinh) kết quả val/test, confusion matrix, mô hình .joblib, run_info.json
├── run_all.sh                   chạy tuần tự bước 1 → 4
└── requirements.txt
```

`features/` và `results/` do script sinh ra, không có trên repo.

## Cài đặt (1 lần)

```bash
git clone https://github.com/kenziehong/human-object-interaction-recognition.git
cd human-object-interaction-recognition
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp        # macOS: XGBoost cần OpenMP
```

- Thiếu `libomp` thì XGBoost báo `Library not loaded: libomp.dylib`.
- Cảnh báo `RuntimeWarning: ... encountered in matmul` là báo nhầm của numpy 2.0 + Accelerate trên macOS (kết quả vẫn đúng); đã tắt trong `config.py`.

## Cách chạy

Mọi kết quả được lưu theo **phiên bản**, không bao giờ ghi đè hay xoá kết quả cũ:
- `features/<fXX>/`: một bộ đặc trưng (đổi tham số đặc trưng hoặc thêm ảnh → phiên bản `f` mới).
- `results/<rXX>/`: một lần chọn mô hình + test, dùng một phiên bản đặc trưng (ghi trong `run_info.json`).
- Script **dừng lại** nếu phiên bản đã tồn tại. Muốn bổ sung vào phiên bản cũ phải truyền `--update` một cách chủ động.
- Mỗi phiên bản có `run_info.json`: thời gian, lệnh đã chạy, tham số, ghi chú (`--note`), số ảnh, cấu hình đặc trưng, lưới siêu tham số.
- **Mỗi phiên bản mới phải được thêm vào bảng [Phiên bản kết quả](#phiên-bản-kết-quả).**

**Chạy toàn bộ bằng 1 lệnh** (ở thư mục gốc repo, khoảng 2 phút trên 519 ảnh):

```bash
bash run_all.sh                          # tạo features/f01 + results/r01, rồi chạy notebook
bash run_all.sh f02 r02 "ghi chú"        # phiên bản khác; nếu features/f02 đã có thì dùng lại
```

Chỉ chạy lại notebook `04_report.ipynb` thì **không** tạo ra kết quả: notebook chỉ đọc `features/` và `results/` do bước 1–3 sinh ra. Repo không chứa hai thư mục này, nên lần đầu phải chạy `run_all.sh` (hoặc bước 1 → 3 bên dưới) trước. Với cùng dữ liệu và thư viện trong `requirements.txt`, kết quả tái lập chính xác (đã kiểm tra: đặc trưng giống từng bit, val/test lệch 0).

Hoặc chạy từng bước trong thư mục `code/` (ví dụ tạo `f02` và `r02`):

```bash
cd code
../.venv/bin/python 01_extract_features.py --version f02 --note "mô tả"                 # bỏ qua nếu dùng lại đặc trưng cũ
../.venv/bin/python 02_train_select.py --feat_version f02 --version r02 --note "mô tả"  # chọn mô hình trên val
../.venv/bin/python 03_final_test.py --version r02                                      # đánh giá test 1 lần
RUN_VERSION=r02 ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 04_report.ipynb  # báo cáo hiển thị r02
```

**Báo cáo kết quả xem ở `code/04_report.ipynb`**: notebook này luôn hiển thị phiên bản mới nhất (hiện là `r01`). Khi có phiên bản mới, sửa giá trị mặc định `VERSION` ở ô đầu rồi Run All (kernel `.venv`). Muốn xem lại phiên bản cũ thì đặt `VERSION` cũ và chạy lại; `results/` chỉ chứa kết quả do script sinh ra, không chứa bản sao notebook hay HTML.

Tùy chọn:
- Thử nhanh: `01_extract_features.py --version f_test --limit 60` (lấy khoảng 10 ảnh mỗi lớp mỗi split).
- Chạy thêm một phần vào phiên bản có sẵn: `02_train_select.py --feat_version f02 --version r02 --update --feats sift color --models svm` (chỉ thay các dòng `--feats × --models`, giữ nguyên các dòng khác). Sau đó chạy `03_final_test.py --version r02 --update`.
- Đổi tham số đặc trưng trong `config.py`, sau đó chạy bước 1 với **phiên bản mới**.

Đầu ra chính (trong `results/<rXX>/`, trừ khi ghi khác):
- `val_results.csv`: macro-F1, balanced accuracy, recall từng lớp, siêu tham số tốt nhất của mọi cấu hình.
- `test_results.csv`, `test_report.txt`: kết quả test của mô hình tốt nhất cho mỗi bộ đặc trưng và cho mỗi mô hình (cột `is_best_on_val` đánh dấu mô hình tốt nhất trên val).
- `confusion_<feature>_<model>.png`, `models/*.joblib`, `test_pred_*.npy`.
- Thời gian chạy (xem mục [Thời gian chạy](#thời-gian-chạy)):
  - `features/<fXX>/timing_extract.csv` (bước 1): `wall_s` (thời gian thực của cả công đoạn, chạy song song), `cpu_s` và `cpu_ms_per_img` (tổng thời gian tính của riêng từng khối đặc trưng).
  - `val_results.csv` (bước 2): `n_grid`, `grid_search_s`, `fit_best_s`, `time_s`.
  - `test_results.csv` (bước 3): `dim`, `fit_s` (train trên train+val), `predict_s`, `predict_ms_per_img`.
  - `run_info.json`: `total_wall_s` của mỗi bước.

> Nếu trong `data/` thiếu ảnh so với `labels.csv`, pipeline in cảnh báo và chỉ dùng ảnh đang có.

## Phiên bản kết quả

Tra cứu nhanh: mỗi dòng là một thư mục; chi tiết lệnh và tham số nằm trong `run_info.json` của thư mục đó. Cột "Thực nghiệm" trỏ tới mục tương ứng trong [Thực nghiệm](#thực-nghiệm).

**Đặc trưng (`features/`)**

| Phiên bản | Ngày | Số ảnh (train/val/test) | Nội dung | Thực nghiệm | Ghi chú |
|---|---|---|---|---|---|
| `f01` | 12/10/2026 | 261 / 42 / 216 | 10 khối: hog, lbp, color, sift, geom, sift_spm, hog_upper, sift_upper, gist, skin | 3 | Tạo bằng `run_all.sh`; có `timing_extract.csv` |

**Kết quả (`results/`)**

| Phiên bản | Ngày | Đặc trưng | Số ảnh (train/val/test) | Nội dung | Kết quả chính | Thực nghiệm | Ghi chú |
|---|---|---|---|---|---|---|---|
| **`r01`** | 12/10/2026 | `f01` | 261 / 42 / 216 | 12 bộ đặc trưng × 4 mô hình, test 15 cấu hình, có đo thời gian | val tốt nhất: sift + XGBoost 0.352; test tốt nhất: geom + XGBoost 0.283 | 1–4 | **Bản mới nhất** (đang hiển thị trong notebook). Cấu hình đã gồm mọi thay đổi của thực nghiệm 1–4 |

Các lần chạy trong thực nghiệm 1–4 không lưu lại file; số liệu của chúng chỉ còn trong mục Thực nghiệm. `r01` tái lập đúng kết quả cuối của thực nghiệm 4.
Phiên bản tiếp theo nên đặt là `r02` (và `f02` khi có đủ dữ liệu).

## Kết quả chạy thử (mẫu nhỏ, 12/10/2026)

Số liệu dưới đây là của phiên bản **`r01`** (`results/r01/`; biểu đồ và bảng đầy đủ trong `code/04_report.ipynb`).

Chạy trên **519/15.187 ảnh** (train 261, val 42, test 216), chỉ để kiểm tra pipeline, **không dùng làm kết quả báo cáo**.
Val chỉ có 42 ảnh và không có `ride`, nên việc chọn mô hình còn rất nhiễu.
Mốc so sánh: đoán ngẫu nhiên 6 lớp cho balanced accuracy ≈ 0.167; đoán toàn `unknown` cho macro-F1 ≈ 0.13.

Val macro-F1 (12 bộ đặc trưng × 4 mô hình):

| Bộ đặc trưng | LogReg | SVM | RF | XGBoost |
|---|---|---|---|---|
| hog | 0.219 | 0.226 | 0.152 | 0.152 |
| lbp | 0.145 | 0.220 | 0.155 | 0.208 |
| color | 0.218 | 0.328 | 0.152 | 0.147 |
| sift | 0.331 | 0.345 | 0.265 | **0.352** |
| geom | 0.154 | 0.219 | 0.255 | 0.266 |
| sift_spm | 0.296 | 0.265 | 0.265 | 0.234 |
| upper | 0.234 | 0.319 | 0.152 | 0.152 |
| gist | 0.223 | 0.199 | 0.152 | 0.227 |
| skin | 0.242 | 0.272 | 0.250 | 0.278 |
| fused | 0.315 | 0.152 | 0.152 | 0.152 |
| combo_local | 0.275 | 0.152 | 0.152 | 0.149 |
| fused_all | 0.323 | 0.152 | 0.152 | 0.265 |

Test (mô hình tốt nhất trên val của mỗi bộ đặc trưng, train lại trên train+val):

| Bộ đặc trưng | Mô hình | Val macro-F1 | Test macro-F1 | Test balanced acc |
|---|---|---|---|---|
| sift | XGBoost (tốt nhất trên val) | **0.352** | 0.173 | 0.178 |
| color | SVM | 0.328 | 0.216 | 0.222 |
| fused_all | LogReg | 0.323 | 0.248 | **0.306** |
| upper | SVM | 0.319 | 0.244 | 0.244 |
| fused | LogReg | 0.315 | 0.254 | 0.264 |
| sift_spm | LogReg | 0.296 | 0.163 | 0.166 |
| skin | XGBoost | 0.278 | 0.213 | 0.221 |
| combo_local | LogReg | 0.275 | 0.232 | 0.252 |
| geom | XGBoost | 0.266 | **0.283** | 0.279 |
| gist | XGBoost | 0.227 | 0.199 | 0.191 |
| hog | SVM | 0.226 | 0.226 | 0.253 |
| lbp | SVM | 0.220 | 0.190 | 0.215 |

Ngoài ra bước 3 còn chạy test cho mô hình tốt nhất của từng loại mô hình: sift + LogReg 0.262, sift + SVM 0.218, sift + RF 0.141 (test macro-F1).

Nhận xét (trên mẫu nhỏ):
- **Val không dự đoán được test**: cấu hình tốt nhất trên val (sift + XGBoost, 0.352) chỉ đạt 0.173 trên test, gần mức đoán ngẫu nhiên. Với 42 ảnh val, chênh lệch 1–2 ảnh đúng/sai đã làm đổi thứ hạng. Cần đủ dữ liệu trước khi kết luận.
- **Phối hợp ổn định hơn đặc trưng đơn lẻ**: `fused`, `fused_all`, `combo_local`, `upper` đều đạt 0.23–0.25 trên test; `fused_all` + LogReg có balanced accuracy cao nhất (0.306).
- **`geom` (8 chiều) mạnh bất ngờ**: test macro-F1 cao nhất (0.283), và có importance trung bình mỗi chiều cao nhất trong RF trên `fused_all` (sau đó là `skin`). Tỉ lệ khung và vị trí người là thông tin mà các đặc trưng ảnh (đã resize) bỏ mất.
- **SVM/RF/XGBoost trên bộ phối hợp lớn thường chỉ đoán `unknown`** (0.152) khi ít dữ liệu; LogReg chịu được số chiều cao tốt hơn.
- `ride` đạt recall 0 ở mọi cấu hình (train 8 ảnh, val 0, test 3).

## Thời gian chạy

Máy: Apple Silicon (arm64), 10 CPU, `n_jobs=-1`, trên mẫu 519 ảnh.

**Bước 1: trích đặc trưng** (`f01`, đo bằng code, `features/f01/timing_extract.csv`). Tổng thời gian thực **16.3 s** (`run_info.json`).

| Khối | CPU (s, cả 519 ảnh) | CPU ms / ảnh |
|---|---|---|
| sift_descriptor (dense SIFT) | 9.49 | 18.3 |
| kmeans K=500 (học codebook, chỉ trên train) | 4.53 | (cố định, chỉ train) |
| gist (32 bộ lọc Gabor) | 2.89 | 5.6 |
| lbp | 1.86 | 3.6 |
| bovw_encode (sift + sift_spm + sift_upper) | 1.45 | 2.8 |
| read_image (đọc ảnh, 2 lượt) | 0.96 | 1.9 |
| hog | 0.72 | 1.4 |
| hog_upper / skin | 0.35 / 0.34 | 0.7 / 0.7 |
| color | 0.11 | 0.2 |
| geom (từ bbox, không đọc ảnh) | 0.01 | 0.02 |

Ước tính cho đủ 15.187 ảnh: khoảng 35 ms CPU / ảnh → ~9 phút CPU, chạy song song khoảng **2–3 phút**. Đây là ước tính, chưa đo.

**Bước 2: chọn mô hình (48 cấu hình)**: `r01`, đo bằng code (`results/r01/val_results.csv`, `run_info.json`). Tổng thời gian thực **69.1 s**.

| Mô hình | Số lần fit trong grid | Grid search (s) | Fit lại cấu hình tốt nhất trên train (s) |
|---|---|---|---|
| LogReg | 48 | 2.9 | 0.8 |
| SVM | 108 | 4.3 | 1.0 |
| RF | 96 | 20.5 | 3.0 |
| XGBoost | 96 | 18.9 | 17.1 |

- Theo bộ đặc trưng (tổng grid search 4 mô hình): `hog` 8.1 s, `fused_all` 6.9 s, `fused` 5.6 s, `color` 5.1 s; rẻ nhất `geom` 0.7 s và `skin` 0.9 s.
- Cấu hình chậm nhất (grid + fit lại): color + XGBoost (5.4 s), gist + XGBoost (4.6 s), hog + RF (4.3 s).
- XGBoost: fit lại 1 lần bằng tất cả CPU (17.1 s cho 12 bộ) tốn ngang cả grid search song song.
- Tổng các cột cộng lại (≈ 68 s) gần bằng thời gian thực; phần còn lại là thời gian nạp dữ liệu và khởi tạo tiến trình.

So với trước khi sửa XGBoost (thực nghiệm 4, đo bằng `time`): toàn bộ bước 2 mất 8 phút 19 s, trong đó XGBoost 461.9 s. Đo riêng 1 lần fit XGBoost trên `fused_all` (~9.300 chiều, 300 cây): `max_bin=256` mất 28.3 s, `max_bin=64` mất 15.0 s.
Với đủ dữ liệu (train lớn gấp ~26 lần), SVM RBF tăng khoảng O(n²), nên bước 2 có thể mất **hàng chục phút đến vài giờ** (ước tính, chưa đo).

**Bước 3: test (15 cấu hình)**: `r01`, tổng thời gian thực **9.0 s** (`run_info.json`).
- Fit trên train+val: tổng 7.4 s; chậm nhất là XGBoost (1.1–1.9 s mỗi cấu hình), LogReg/SVM ≤ 0.21 s.
- Dự đoán: tổng 0.15 s cho 15 × 216 ảnh, ≤ 0.21 ms/ảnh. Thời gian chạy thật vì vậy chủ yếu nằm ở trích đặc trưng (~35 ms CPU/ảnh).

**Bước 4: notebook**: khoảng 16 s (ước tính = tổng `run_all.sh` 110.8 s trừ bước 1–3; gồm cả thời gian khởi động Python).

**Cả pipeline** (`run_all.sh`, đo bằng `time`): **1 phút 51 s**.

## Thực nghiệm

Tất cả chạy trên mẫu **519 ảnh** (train 261, val 42, test 216). Thời gian đo trên Apple Silicon, 10 CPU.

1. **Bản đầu: 4 đặc trưng cơ bản × 3 mô hình** (12/10/2026)
   - Vấn đề / lý do: yêu cầu đề bài (3 loại đặc trưng cơ bản, 3 mô hình ML cơ bản); người làm đồ án chọn 4 đặc trưng.
   - Thay đổi:
     - Đặc trưng: HOG (64×128, 3780-d), LBP (P=24, R=3, uniform, lưới 3×3 để giữ vị trí, 234-d), Color HSV (8×8×8, nửa trên/dưới, 1024-d), Dense SIFT (bước 8 px, size 16) + BoVW (K=500, MiniBatchKMeans chỉ học trên train, chuẩn hóa Hellinger, 500-d), cộng `fused` = nối 4 khối.
     - Mô hình: LogReg, SVM RBF (PCA 256 cho HOG/fused để giảm chiều), RF.
     - Cân bằng lớp bằng `class_weight` (không oversample, để giữ phân bố thật). Chọn siêu tham số bằng `GridSearchCV` + `PredefinedSplit` (train → val), scoring macro-F1.
   - Kết quả (mốc ban đầu):
     - Val tốt nhất: SIFT + SVM 0.345.
     - Test tốt nhất: Fused + LogReg 0.254.
     - Thời gian: chưa đo.
   - Nhận xét: RF thường chỉ đoán `unknown` khi mỗi lớp chỉ có vài chục ảnh.
   - Phiên bản: không còn file (xem bảng Phiên bản kết quả).
2. **Thêm XGBoost** (12/10/2026)
   - Vấn đề / lý do: người làm đồ án yêu cầu; XGBoost thường mạnh với dữ liệu dạng bảng.
   - Thay đổi:
     - Cân bằng lớp bằng `sample_weight`, vì XGBoost multi-class không có `class_weight`.
     - Lưới: `n_estimators` {300, 600}, `max_depth` {4, 6}, `learning_rate` {0.05, 0.1}; cố định `subsample` 0.8, `colsample_bytree` 0.5, `max_bin` 256 (mặc định).
   - Kết quả:
     - Val tốt nhất không đổi: 0.345 → 0.345 (SIFT + SVM). Tốt nhất của XGBoost: SIFT + XGBoost 0.239; các bộ khác 0.15–0.21.
     - Test không đổi: XGBoost không được chọn cho bộ đặc trưng nào, nên không chạy test.
     - Thời gian: chưa đo.
   - Nhận xét: với 8–22 ảnh mỗi lớp ở train, mô hình cây gần như chỉ đoán `unknown`.
   - Phiên bản: không còn file.
3. **Thêm 5 đặc trưng và 2 cách phối** (12/10/2026)
   - Vấn đề / lý do: dựa trên README dữ liệu.
     - `talk_on_phone` / `eat` có vật nhỏ nằm gần đầu, dễ bị "chìm" khi tính đặc trưng trên cả crop.
     - Vật của `ride` chỉ ~47% nằm trong crop, nên phải dựa vào bối cảnh và dáng người.
     - BoVW gốc mất thông tin vị trí.
     - Resize crop làm mất tỉ lệ khung người.
   - Thay đổi:
     - `geom` (8-d: tỉ lệ/kích thước bbox, mức nới thực tế 4 phía).
     - `sift_spm` (spatial pyramid 1x1 + 2x2, trọng số 0.5/0.5).
     - `upper` (40% phía trên: HOG 64×64 + BoVW).
     - `gist` (Gabor kernel 5/9/15/21 × 8 hướng, lưới 4×4).
     - `skin` (YCrCb Cr∈[133,173], Cb∈[77,127], lưới 4×4 + trọng tâm).
     - Tổ hợp `combo_local` (sift_spm + sift_upper + color + geom + skin; nhắm vào vật nhỏ + vị trí + bối cảnh) và `fused_all` (mọi khối, dùng `sift_spm` thay cho `sift`).
     - PCA cho SVM chuyển từ danh sách cố định sang tự động khi số chiều > 1500.
     - Cách đánh giá: `03_final_test.py` chạy test thêm cho mô hình tốt nhất của từng loại mô hình.
   - Kết quả:
     - Val tốt nhất không đổi: 0.345 → 0.345 (SIFT + SVM). Đặc trưng mới tốt nhất trên val: `fused_all` + LogReg 0.323, `upper` + SVM 0.319.
     - Test tốt nhất: 0.254 → **0.267** (`geom` + RF). Ngoài ra `fused_all` + LogReg 0.248 có balanced accuracy cao nhất (0.306); `upper` + SVM 0.244.
     - Thời gian bước 2 (48 cấu hình): 8 phút 19 s, đo bằng `time`; XGBoost chiếm 461.9 s.
   - Nhận xét: `geom` (8 chiều) có importance trung bình mỗi chiều cao nhất trong RF trên `fused_all`, tiếp theo là `skin`.
   - Phiên bản: không còn file; đặc trưng cùng tham số với `f01`.
4. **Sửa XGBoost chạy chậm** (12/10/2026)
   - Vấn đề / lý do:
     - XGBoost chiếm 461.9/499 s của bước 2, dù chỉ có 261 ảnh train.
     - Đo riêng 1 lần fit trên `fused_all` (~9.300 chiều, 300 cây): 28.3 s.
     - Với đủ dữ liệu, riêng XGBoost có thể chạy hàng giờ (ước tính, chưa đo).
   - Thay đổi (`code/models.py`):
     - `max_bin` 256 → 64.
     - Thêm `PCA(256)` trước XGBoost khi bộ đặc trưng > 1500 chiều (áp dụng cho `hog`, `sift_spm`, `upper`, `fused`, `combo_local`, `fused_all`).
   - Kết quả:
     - Thời gian: XGBoost cả 12 bộ đặc trưng 461.9 s → 46 s; riêng `max_bin=64` (chưa PCA) giảm 1 lần fit từ 28.3 s xuống 15.0 s.
     - Val tốt nhất: 0.345 → **0.352** (SIFT + XGBoost, trước đó 0.239). `geom` + XGBoost 0.266 vượt `geom` + RF 0.255.
     - Test tốt nhất: 0.267 → **0.283** (`geom` + XGBoost, thay cho `geom` + RF). Nhưng cấu hình tốt nhất trên val (SIFT + XGBoost) chỉ đạt **0.173** trên test.
   - Nhận xét:
     - Bộ `sift` chỉ có 500 chiều nên **không bị PCA**: mức tăng của SIFT + XGBoost đến từ `max_bin=64`, cộng thêm độ nhiễu của val 42 ảnh, không phải từ PCA.
     - Val nhỏ dễ chọn nhầm: tốt nhất trên val lại gần mức đoán ngẫu nhiên trên test.
   - Phiên bản: kết quả cuối được tái lập trong `r01` (bước 2 mất 69.1 s).

Kết luận tạm: trên mẫu nhỏ, các tổ hợp (`fused`, `fused_all`, `combo_local`) ổn định hơn đặc trưng đơn lẻ; `geom` là đặc trưng rẻ mà hiệu quả nhất; LogReg ổn nhất với dữ liệu ít và nhiều chiều. **Chưa có kết luận nào đáng tin cho tới khi chạy đủ dữ liệu.**

### Bài học kinh nghiệm

- **XGBoost chậm theo số cột, không phải theo số ảnh.** Thuật toán `hist` dựng histogram cho từng cột ở mỗi vòng, và multi-class 6 lớp nghĩa là 6 cây mỗi vòng. Vì vậy với bộ 9.300 chiều, 261 ảnh vẫn mất khoảng 28 s/lần fit.
  - Cách xử lý: **giảm chiều bằng PCA (256) trước XGBoost** khi > 1500 chiều, và giảm `max_bin` xuống 64.
  - Cái giá: cây không còn tách trực tiếp trên từng chiều gốc của BoVW/HOG nữa, mà trên thành phần PCA.
  - Áp dụng tương tự cho SVM RBF (cũng PCA 256).
- **Không chạy song song lồng nhau**: `GridSearchCV(n_jobs=-1)` đã song song theo cấu hình, nên mô hình bên trong (RF/XGBoost) để `n_jobs=1`. Chỉ khi fit một mô hình đơn lẻ mới cho dùng hết CPU (`use_all_cores` trong `models.py`).
- **XGBoost không có `class_weight`**: phải truyền `sample_weight=compute_sample_weight("balanced", y)` vào `fit` (qua `clf__sample_weight` trong Pipeline). Train phải có đủ cả 6 lớp, nếu không XGBoost báo lỗi nhãn.
- **Val nhỏ thì không chọn mô hình được**: với 42 ảnh, cấu hình tốt nhất trên val (0.352) chỉ đạt 0.173 trên test. Cần đủ dữ liệu, hoặc chuyển sang cross-validation theo ảnh.
- **Mô hình cây / SVM trên bộ phối hợp lớn dễ sụp về đoán toàn `unknown`** (macro-F1 0.152) khi mỗi lớp chỉ có vài chục ảnh; LogReg ổn định hơn.
- **Chọn mô hình phải có luật phá hoà**: với val nhỏ, nhiều cấu hình có cùng macro-F1 (ví dụ sift + RF và sift_spm + RF cùng 0.26462). Sắp xếp không ổn định thì mỗi lần chạy có thể chọn cấu hình khác; `rank_results` trong `models.py` xét macro-F1 → balanced acc → tên.

## Cần làm

**Trạng thái hiện tại (để tiếp tục lần sau):**
- `.venv` đã cài đủ thư viện, `libomp` đã cài.
- `features/f01/` có đủ 10 khối đặc trưng cho 519 ảnh hiện có. **Không cần chạy lại bước 1**, trừ khi có thêm ảnh hoặc đổi tham số đặc trưng (khi đó tạo `f02`).
- `results/r01/` là **kết quả đầy đủ mới nhất** trên 519 ảnh (mô hình, dự đoán, confusion, thời gian); biểu đồ xem trong `code/04_report.ipynb`.
- Việc tiếp theo cần **dữ liệu đầy đủ**: khi có, chạy `bash run_all.sh f02 r02 "đủ 15.187 ảnh"`.
  Sau đó sửa `VERSION` mặc định trong ô đầu notebook thành `'r02'`, và thêm dòng `f02`, `r02` vào bảng Phiên bản kết quả. Bước 2 với đủ dữ liệu có thể rất lâu: nên chạy thử trước một phần (`02_train_select.py --feat_version f02 --version r02_thu --feats geom sift --models logreg svm`) để ước tính thời gian.

Việc bắt buộc, làm trước:
- [ ] **Lấy đủ dữ liệu**: hiện chỉ có 519/15.187 crop trong `data/vcoco_6class_crops/`. Kiểm tra bằng `find data/vcoco_6class_crops/train -name '*.jpg' | wc -l`, phải ra 6803.
- [x] Chạy bước 1 → 4 thành phiên bản `f01` / `r01` (`run_all.sh`).
- [x] Thêm phần "Thời gian chạy" vào `04_report.ipynb` (mục 5).
- [ ] Sau khi có đủ dữ liệu: cập nhật các bảng kết quả, bảng thời gian và nhận xét trong README; viết lại phần "Nhận xét gợi ý" (mục 7) trong notebook theo kết quả thật.

Cải thiện mô hình và đặc trưng (chưa làm):
- [ ] **Fisher Vector / VLAD** thay cho BoVW để mã hoá SIFT mạnh hơn.
- [ ] **Embedding CNN pretrain ImageNet** (ResNet50, EfficientNet; đã có code mẫu trong `code/extract_image_and_embedding.ipynb`) đưa qua 4 mô hình ML, làm mốc so sánh. Không dùng backbone pretrain trên COCO, vì nó đã thấy ảnh test.
- [ ] Thử tham số đặc trưng: `BOVW_K` (200/1000), spatial pyramid thêm mức 4×4, `UPPER_FRAC` (0.3/0.5), mức nới SIFT, chuẩn hoá từng khối trước khi nối. Lý do: các giá trị hiện tại được chọn theo kinh nghiệm, chưa thử; riêng chuẩn hoá từng khối vì khi nối, khối nhiều chiều (HOG 3780-d) lấn át khối ít chiều (`geom` 8-d), trong khi `geom` lại là đặc trưng mạnh nhất.
- [ ] **Late fusion**: lấy trung bình xác suất của các mô hình tốt nhất trên từng đặc trưng (cần `SVC(probability=True)`), so sánh với nối đặc trưng (early fusion). Lý do: SVM/RF/XGBoost trên bộ nối lớn hay sụp về đoán toàn `unknown`, trong khi từng đặc trưng riêng lẻ vẫn học được; gộp ở mức xác suất tránh được vấn đề số chiều.
- [ ] Chỉnh ngưỡng quyết định cho lớp `unknown` trên val để tăng recall các lớp hành động. Lý do: `unknown` chiếm ~65%, và trên test của `r01`, recall `unknown` là 0.56–1.0 trong khi recall các lớp hành động chỉ 0–0.5.
- [ ] Nếu val vẫn nhiễu: chọn siêu tham số bằng `StratifiedGroupKFold` trên train+val (nhóm theo `image_id`) thay cho 1 tập val cố định.
- [ ] Cờ lọc crop nhỏ `MIN_SIDE` đã có trong `config.py` nhưng **chưa được dùng**: cần nối vào `01_extract_features.py` (chỉ lọc ở train) nếu muốn thử. Lý do: 16 crop có cạnh ngắn < 32 px (README dữ liệu), khi phóng lên 64×128 / 128×256 thì HOG, LBP gần như chỉ còn nhiễu.
- [ ] Nếu bước 2 quá chậm với đủ dữ liệu: thu hẹp lưới SVM/XGBoost, hoặc chạy theo nhóm bằng `--feats ... --models ...`. Lý do: SVM RBF tăng khoảng O(n²) theo số ảnh, train đủ lớn gấp ~26 lần mẫu hiện tại (ước tính hàng chục phút đến vài giờ).

## Tài liệu tham khảo

- Li et al., *Disentangled Pre-training for Human-Object Interaction Detection* (DP-HOI), CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Disentangled_Pre-training_for_Human-Object_Interaction_Detection_CVPR_2024_paper.pdf

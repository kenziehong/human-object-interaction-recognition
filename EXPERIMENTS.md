# Thực nghiệm

Nhật ký đầy đủ các thay đổi đặc trưng / mô hình nhằm cải thiện kết quả, kèm lý do và số đo. `README.md` chỉ giữ thực nghiệm mới nhất; phiên bản `fXX` / `rXX` tra trong mục Phiên bản kết quả của [README](README.md#phiên-bản-kết-quả).

Mục 1–4 chạy trên mẫu **519 ảnh** V-COCO (train 261, val 42, test 216); mục 5 trở đi chạy trên bộ gộp 4 nguồn (12.833 ảnh, train 9.069 / val 1.878 / test 1.886). Thời gian đo trên Apple Silicon, 10 CPU.

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
   - Phiên bản: không còn file (xem bảng [Phiên bản kết quả](README.md#phiên-bản-kết-quả)).
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

5. **Chuyển sang bộ gộp 4 nguồn, giảm `unknown`, chỉ giữ sift + hog × LogReg** (24/10/2026)
   - Vấn đề / lý do:
     - `r01` chỉ có 519 ảnh (val 42): chọn mô hình rất nhiễu (val tốt nhất 0.352 nhưng test 0.173).
     - Bộ gộp `dataset_6class_crops` có 22.366 ảnh, nhiều ảnh hành động hơn hẳn bộ V-COCO đang có.
     - `unknown` chiếm 56%; giảm xuống 3.000 để các lớp cân bằng hơn và chạy nhanh hơn.
     - Chạy đủ 12 bộ × 4 mô hình ước tính 1–1,5 giờ; người làm đồ án chọn chỉ giữ 2 đặc trưng cơ bản bổ sung cho nhau (SIFT: chi tiết vật; HOG: dáng người) và 1 mô hình (LogReg, ổn định nhất ở `r01`). `geom` bị loại vì dựa vào bbox trong dữ liệu chứ không phải đặc trưng ảnh truyền thống.
   - Thay đổi:
     - Dữ liệu: `vcoco_6class_crops/labels.csv` → `train_test_split/labels_split_unk3000.csv` (chia 70/15/15 theo ảnh gốc, phân tầng lớp × nguồn, seed 2026; giữ 3.000 `unknown` đầu sau khi xáo).
     - Đặc trưng dùng ở bước 2: 12 bộ → `sift`, `hog`. Mô hình: 4 → LogReg (lưới C {0.01, 0.1, 1, 10} không đổi). Tham số đặc trưng không đổi.
   - Kết quả (`f02` / `r02`, 12.833 ảnh):
     - sift + LogReg: val 0.331 → **0.430**, test 0.262 → **0.434** (`r01` → `r02`).
     - hog + LogReg: val 0.219 → 0.362; test 0.387 (`r01` không chạy test cho hog + LogReg).
     - Chênh lệch val–test ≤ 0.03 (ở `r01` tới 0.18).
     - Thời gian: bước 1 247 s, bước 2 16.3 s, bước 3 4.4 s.
   - Nhận xét / tác dụng phụ:
     - **Không so trực tiếp với `r01`**: khác dữ liệu, khác test, và test của `r02` đã giảm `unknown`. Phần tăng chủ yếu đến từ có nhiều dữ liệu hơn (train 261 → 9.069) và phân bố lớp cân bằng hơn, không phải từ thay đổi mô hình.
     - C tốt nhất là 0.01 (biên dưới của lưới) cho cả 2 đặc trưng, nên lưới C chưa đủ rộng.
     - Bỏ các bộ phối và `geom` nên không còn mốc "phối có hơn đơn lẻ không" trên dữ liệu lớn.
   - Phiên bản: `f02`, `r02` (lần chạy dở dang: `f02_interrupted`, `r02_interrupted`, không dùng).

6. **Thêm Random Forest vào `r02`** (24/10/2026)
   - Vấn đề / lý do: người làm đồ án muốn so LogReg với một mô hình phi tuyến trên cùng đặc trưng `sift`, `hog` của `f02`. Ước tính trước khi chạy: 5–7 phút, chủ yếu là hog + RF.
   - Thay đổi: bổ sung RF vào `r02` bằng `--update` (người làm đồ án đồng ý). Lưới RF giữ nguyên như `r01`: `n_estimators` {300, 500}, `max_depth` {None, 30}, `min_samples_leaf` {1, 3}, `class_weight="balanced_subsample"`.
   - Kết quả:
     - sift + RF: val 0.415, test **0.432** (sift + LogReg: val 0.430, test 0.434).
     - hog + RF: val 0.339 (hog + LogReg: 0.362); không chạy test.
     - Cả 2 đặc trưng chọn cùng cấu hình: 300 cây, `max_depth=None`, `min_samples_leaf=3`.
     - Thời gian bước 2: sift + RF 24.5 s, hog + RF 237.5 s (LogReg: 1.7 s và 14.6 s).
   - Nhận xét / tác dụng phụ:
     - RF **không hơn** LogReg về macro-F1, mà tốn gấp 14–16 lần thời gian ở bước 2.
     - RF đổi kiểu lỗi: tăng recall `hold` (0.237 → 0.322), `ride`, `unknown`, nhưng giảm mạnh `eat/drink` (0.370 → 0.197) và `talk_on_phone` (0.367 → 0.293) trên test. Hai mô hình sai ở các lớp khác nhau, nên late fusion LogReg + RF có thể có ích (chưa thử).
     - `min_samples_leaf=3` thắng 1 ở cả 2 đặc trưng: lá lớn hơn (ít khớp quá mức) tốt hơn.
   - Phiên bản: `r02` (cập nhật bằng `--update`; `run_info.json` ghi cả 2 lần chạy bước 2 và 3).

Kết luận tạm: trên mẫu nhỏ, các tổ hợp (`fused`, `fused_all`, `combo_local`) ổn định hơn đặc trưng đơn lẻ; `geom` là đặc trưng rẻ mà hiệu quả nhất; LogReg ổn nhất với dữ liệu ít và nhiều chiều. **Chưa có kết luận nào đáng tin cho tới khi chạy đủ dữ liệu.**

## Kết quả chạy thử trên mẫu nhỏ (`r01`, thực nghiệm 1–4, 12/10/2026)

Số liệu dưới đây là của phiên bản **`r01`** (`data/results/r01/`; biểu đồ và bảng đầy đủ trong `code/04_report.ipynb`).

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

### Thời gian chạy `f01` / `r01` (mẫu 519 ảnh, 12/10/2026)

Máy: Apple Silicon (arm64), 10 CPU, `n_jobs=-1` (như mục [Thời gian chạy](README.md#thời-gian-chạy) của README).

**Bước 1: trích đặc trưng** (`f01`, đo bằng code, `data/features/f01/timing_extract.csv`). Tổng thời gian thực **16.3 s** (`run_info.json`).

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

**Bước 2: chọn mô hình (48 cấu hình)**: `r01`, đo bằng code (`data/results/r01/val_results.csv`, `run_info.json`). Tổng thời gian thực **69.1 s**.

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

## Bài học kinh nghiệm

- **XGBoost chậm theo số cột, không phải theo số ảnh.** Thuật toán `hist` dựng histogram cho từng cột ở mỗi vòng, và multi-class 6 lớp nghĩa là 6 cây mỗi vòng. Vì vậy với bộ 9.300 chiều, 261 ảnh vẫn mất khoảng 28 s/lần fit.
  - Cách xử lý: **giảm chiều bằng PCA (256) trước XGBoost** khi > 1500 chiều, và giảm `max_bin` xuống 64.
  - Cái giá: cây không còn tách trực tiếp trên từng chiều gốc của BoVW/HOG nữa, mà trên thành phần PCA.
  - Áp dụng tương tự cho SVM RBF (cũng PCA 256).
- **Không chạy song song lồng nhau**: `GridSearchCV(n_jobs=-1)` đã song song theo cấu hình, nên mô hình bên trong (RF/XGBoost) để `n_jobs=1`. Chỉ khi fit một mô hình đơn lẻ mới cho dùng hết CPU (`use_all_cores` trong `models.py`).
- **XGBoost không có `class_weight`**: phải truyền `sample_weight=compute_sample_weight("balanced", y)` vào `fit` (qua `clf__sample_weight` trong Pipeline). Train phải có đủ cả 6 lớp, nếu không XGBoost báo lỗi nhãn.
- **Val nhỏ thì không chọn mô hình được**: với 42 ảnh, cấu hình tốt nhất trên val (0.352) chỉ đạt 0.173 trên test. Cần đủ dữ liệu, hoặc chuyển sang cross-validation theo ảnh.
- **Mô hình cây / SVM trên bộ phối hợp lớn dễ sụp về đoán toàn `unknown`** (macro-F1 0.152) khi mỗi lớp chỉ có vài chục ảnh; LogReg ổn định hơn.
- **`class_weight="balanced_subsample"` chưa đủ để RF quan tâm lớp nhỏ**: ở `r02`, RF vẫn thiên về `hold`/`ride`/`unknown` (lớp nhiều ảnh) và bỏ rơi `eat/drink`, `talk_on_phone`. Với đặc trưng nhiều chiều (hog 3780-d), lệch còn mạnh hơn. LogReg cân bằng giữa các lớp tốt hơn.
- **Giá trị tốt nhất nằm ở biên lưới thì lưới chưa đủ rộng**: ở `r02`, C = 0.01 (nhỏ nhất) thắng ở cả sift và hog. Cần mở rộng lưới về phía đó trước khi kết luận.
- **Ước tính thời gian từ mẫu nhỏ có thể thấp hơn thực tế**: ước tính bước 1 cho 12.833 ảnh từ số đo 519 ảnh là 2–4 phút, thực tế 4 phút 7 s; ms/ảnh của SIFT và BoVW tăng 3.5–11 lần khi chạy song song lâu (nguyên nhân chưa xác định, xem [Thời gian chạy](README.md#thời-gian-chạy)). Nên chạy thử trên vài nghìn ảnh trước khi ước tính cho lần chạy lớn.
- **Chọn mô hình phải có luật phá hoà**: với val nhỏ, nhiều cấu hình có cùng macro-F1 (ví dụ sift + RF và sift_spm + RF cùng 0.26462). Sắp xếp không ổn định thì mỗi lần chạy có thể chọn cấu hình khác; `rank_results` trong `models.py` xét macro-F1 → balanced acc → tên.

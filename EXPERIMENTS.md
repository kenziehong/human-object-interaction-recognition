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

7. **4 đặc trưng cơ bản × 4 mô hình, đủ chỉ số và đo RAM** (24/10/2026)
   - Vấn đề / lý do: `r02` mới thử 2 đặc trưng (sift, hog) × 2 mô hình (LogReg, RF) trên dữ liệu đầy đủ; chưa biết `lbp`, `color` và SVM, XGBoost làm được gì khi có đủ dữ liệu (kết quả trên mẫu 519 ảnh không đáng tin). Đồng thời đánh giá chưa đủ chỉ số (thiếu accuracy, precision, confusion số ảnh) và chưa đo RAM.
   - Thay đổi:
     - Chạy `hog`, `lbp`, `color`, `sift` × LogReg, SVM, RF, XGBoost trên `f02`. Lưới giữ nguyên như `r01`.
     - Cách đánh giá: `eval_metrics` (models.py) ghi accuracy, macro precision, weighted F1, P/R/F1 từng lớp cho cả val và test; confusion matrix dạng số ảnh (csv + png); lưu `val_pred_*.npy`.
     - Bước 3 **test mọi cấu hình** (trước đây chỉ test cấu hình tốt nhất mỗi đặc trưng / mỗi mô hình); cấu hình chọn vẫn theo val, riêng cho từng nhóm (`group`).
     - Đo RAM đỉnh, CPU, dung lượng mô hình (`code/resources.py`).
   - Kết quả:
     - Cấu hình chọn theo val: **sift + SVM** (C = 1), val 0.480, test **0.464** (`r02`: sift + LogReg, test 0.434; +0.030).
     - SVM tốt nhất trên val cho sift, hog, lbp; XGBoost cho color (color + XGB val 0.458, color + LogReg chỉ 0.335).
     - hog + SVM: val 0.445 (thứ 4) nhưng test cao nhất 0.483; recall `hold` 0.466 so với 0.274 của sift + SVM.
     - Tổng thời gian 26 phút 56 s (ước tính 17–25 phút + 3–6 phút do test 16 cấu hình); RAM đỉnh bước 2: 7.8 GB (hog + RF).
   - Nhận xét / tác dụng phụ:
     - Cải thiện đến từ **mô hình phi tuyến** (SVM RBF), không phải đặc trưng mới: cùng sift, LogReg 0.434 → SVM 0.464 trên test.
     - Lỗi chính vẫn là nhầm giữa 4 lớp "tay cầm vật" (46% lỗi, `r02` 48%).
     - Val đáng tin (Spearman val–test 0.86 trên 16 cấu hình), nên việc chọn theo val ổn; nhưng 0.035 chênh lệch giữa sift + SVM và hog + SVM trên val đảo chiều trên test, nên các cấu hình trong khoảng ~0.03 nên coi là ngang nhau.
     - RF không đáng dùng tiếp: kém nhất trong 3 mô hình phi tuyến, mô hình 98–115 MB.
     - Nhiều siêu tham số tốt nhất nằm ở biên lưới (LogReg C nhỏ nhất, SVM C lớn nhất ở hog, XGB 600 cây).
   - Phiên bản: `r03` (đặc trưng `f02`).

8. **CLIP ViT-B/32: zero-shot với ngưỡng T và linear probe (mốc so sánh)** (24/10/2026)
   - Vấn đề / lý do: ở `r03`, 46% lỗi của cấu hình tốt nhất là nhầm giữa 4 lớp "tay cầm vật": đặc trưng thủ công thấy có tay, có vật nhỏ nhưng không biết vật gì. CLIP mã hoá ngữ nghĩa vật / hành động. Người làm đồ án đề xuất zero-shot theo ý tưởng ngưỡng T chọn Reliable Person Query của DP-HOI, với T tham khảo 0.7 và 0.8.
   - Thay đổi:
     - Khối đặc trưng `clip` (`code/clip_feat.py`, `01_extract_features.py --clip`): embedding ảnh CLIP ViT-B/32 `openai`, 512 chiều, chuẩn hoá L2; ảnh crop được **đệm thành hình vuông** (tiền xử lý mặc định cắt giữa sẽ mất đầu/chân).
     - Mô hình `clip_zs` (`models.ClipZeroShot`): 5 prompt "a photo of a person {holding something / riding / eating or drinking / talking on the phone / using a computer}"; xác suất = softmax(100 × cosine); lớp cao nhất, nếu xác suất cao nhất < T thì `unknown`. **T đặt trên xác suất softmax** (cosine thô của CLIP chỉ ~0.2–0.3 nên T = 0.7/0.8 không có nghĩa với cosine). Lưới T ∈ {0, 0.5, 0.6, 0.7, 0.8, 0.9}, chọn trên val.
     - Linear probe: `clip` × LogReg, SVM, RF, XGB như các đặc trưng khác. Kết quả CLIP thuộc nhóm `clip`, chọn tốt nhất riêng với nhóm thủ công.
     - Lưu điểm val của từng giá trị trong lưới (`grid_*.csv`).
   - Kết quả (`f03`, `r04`):
     - Zero-shot: val chọn **T = 0.6** (0.605); T = 0.7: 0.604, T = 0.8: 0.599, không ngưỡng: 0.562. Test **0.581** (accuracy 0.547).
     - Linear probe: **clip + SVM** val 0.878, test **0.889** (accuracy 0.892); clip + LogReg test 0.876.
     - So với đặc trưng thủ công tốt nhất theo val (sift + SVM, test 0.464): zero-shot +0.117, linear probe +0.425.
     - Thời gian: trích CLIP 151.7 s cho 12.833 ảnh trên CPU; cả pipeline 15 phút 37 s.
   - Nhận xét / tác dụng phụ:
     - **Ngưỡng T có ích nhưng không nhạy**: có ngưỡng tăng val 0.562 → 0.605; mọi T từ 0.5 đến 0.9 cho 0.595–0.605. Hai giá trị tham khảo 0.7 và 0.8 không hơn 0.6.
     - **Zero-shot hỏng ở 2 lớp định nghĩa bằng phủ định**: 65% ảnh `unknown` bị đoán là `ride` (không có prompt cho `unknown`, softmax luôn chọn 1 trong 5 lớp); 58% ảnh `hold` bị gán `unknown` ("holding something" quá mơ hồ). Lớp có vật rõ thì tốt (`work_on_computer` F1 0.899).
     - Linear probe giải quyết đúng 2 lớp đó (`hold` recall 0.848, `unknown` 0.902): cần nhãn để học ranh giới "cầm vật nhưng không làm 3 hành động kia".
     - So sánh với DP-HOI chỉ là mô phỏng lỏng: ở bài báo T = 0.9 là ngưỡng điểm phát hiện người để chọn RPQ, ở đây T là ngưỡng tin cậy hành động.
     - **Rủi ro dữ liệu**: CLIP train trên dữ liệu web không công bố, có thể chứa ảnh COCO (gồm ảnh test V-COCO). Kết quả CLIP chỉ là mốc so sánh, không thay kết quả chính của đặc trưng thủ công.
   - Phiên bản: `f03` (f02 + khối `clip`), `r04`.

Kết luận tạm: trên mẫu nhỏ, các tổ hợp (`fused`, `fused_all`, `combo_local`) ổn định hơn đặc trưng đơn lẻ; `geom` là đặc trưng rẻ mà hiệu quả nhất; LogReg ổn nhất với dữ liệu ít và nhiều chiều. **Chưa có kết luận nào đáng tin cho tới khi chạy đủ dữ liệu.**

## Kết quả `r03` (4 đặc trưng cơ bản × 4 mô hình, 24/10/2026)

Chạy trên **12.833 ảnh** (`f02`, `labels_split_unk3000.csv`: train 9.069, val 1.878, test 1.886), 4 đặc trưng cơ bản (`hog`, `lbp`, `color`, `sift`) × 4 mô hình (LogReg, SVM, RF, XGBoost), **test cả 16 cấu hình**. Cấu hình được chọn là cấu hình tốt nhất **trên val**: **sift + SVM**. Số test của các cấu hình khác chỉ để báo cáo. Test có `unknown` đã giảm (23% thay vì 56%), nên precision các lớp hành động cao hơn so với phân bố thật. Biểu đồ đầy đủ trong `code/04_report.ipynb`.

Xếp theo val macro-F1. Accuracy chỉ để tham khảo; macro recall = balanced accuracy. RAM là RAM đỉnh khi grid search ở bước 2 (cận trên, gồm nền worker; xem Thời gian chạy `r03` bên dưới).

| Đặc trưng | Mô hình | Siêu tham số tốt nhất | Val macro-F1 | Test macro-F1 | Test accuracy | Test macro precision | Test macro recall | Test weighted F1 | Bước 2 (s) | RAM bước 2 (MB) | Mô hình (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **sift** | **SVM** | C = 1, γ = scale | **0.480** | 0.464 | 0.488 | 0.460 | 0.478 | 0.484 | 125.5 | 5.670 | 41.45 |
| color | XGB | lr 0.1, depth 4, 600 cây | 0.458 | 0.432 | 0.481 | 0.431 | 0.438 | 0.472 | 215.4 | 6.848 | 5.81 |
| sift | XGB | lr 0.1, depth 6, 300 cây | 0.449 | 0.447 | 0.477 | 0.448 | 0.452 | 0.472 | 62.5 | 5.655 | 5.50 |
| hog | SVM | C = 10, γ = scale | 0.445 | **0.483** | **0.526** | **0.494** | **0.479** | **0.521** | 41.6 | 7.106 | 26.39 |
| sift | LogReg | C = 0.01 | 0.430 | 0.434 | 0.457 | 0.432 | 0.451 | 0.455 | 0.9 | 5.409 | 0.04 |
| lbp | SVM | C = 1, γ = scale | 0.422 | 0.409 | 0.441 | 0.410 | 0.420 | 0.439 | 49.3 | 5.660 | 19.75 |
| sift | RF | 300 cây, depth None, leaf 3 | 0.415 | 0.432 | 0.468 | 0.455 | 0.437 | 0.456 | 34.8 | 4.933 | 114.64 |
| lbp | XGB | lr 0.1, depth 6, 600 cây | 0.409 | 0.401 | 0.453 | 0.423 | 0.402 | 0.440 | 52.9 | 5.829 | 13.06 |
| lbp | LogReg | C = 0.1 | 0.401 | 0.366 | 0.397 | 0.361 | 0.386 | 0.386 | 1.7 | 4.689 | 0.02 |
| hog | XGB | lr 0.05, depth 4, 600 cây | 0.399 | 0.444 | 0.479 | 0.443 | 0.454 | 0.470 | 60.5 | 6.629 | 9.95 |
| color | RF | 300 cây, depth None, leaf 3 | 0.393 | 0.376 | 0.440 | 0.398 | 0.388 | 0.414 | 82.3 | 5.972 | 105.71 |
| color | SVM | C = 1, γ = scale | 0.378 | 0.359 | 0.402 | 0.357 | 0.370 | 0.392 | 237.1 | 6.412 | 85.50 |
| lbp | RF | 300 cây, depth None, leaf 3 | 0.372 | 0.347 | 0.413 | 0.388 | 0.356 | 0.391 | 50.0 | 6.120 | 107.09 |
| hog | LogReg | C = 0.01 | 0.362 | 0.387 | 0.411 | 0.387 | 0.392 | 0.415 | 15.0 | 4.513 | 0.27 |
| hog | RF | 300 cây, depth None, leaf 3 | 0.339 | 0.353 | 0.431 | 0.445 | 0.359 | 0.403 | 295.7 | 7.837 | 98.42 |
| color | LogReg | C = 0.01 | 0.335 | 0.327 | 0.355 | 0.327 | 0.333 | 0.357 | 4.6 | 6.153 | 0.07 |

Nhận xét:
- **Mô hình phi tuyến giúp nhiều**: sift + SVM đạt test **0.464**, so với 0.434 của sift + LogReg (cấu hình tốt nhất của `r02`). SVM là mô hình tốt nhất trên val cho 3/4 đặc trưng (sift, hog, lbp); XGBoost tốt nhất cho `color`.
- **`color` chỉ có ích với mô hình cây tăng cường**: color + LogReg 0.335 (kém nhất), color + XGB 0.458 (thứ 2 trên val). Histogram màu không tách tuyến tính được, nhưng XGBoost tìm được ngưỡng trên từng bin màu (bối cảnh: đường, cỏ, bàn).
- **RF kém nhất trong 3 mô hình phi tuyến** ở cả 4 đặc trưng, lại nặng nhất (98–115 MB mỗi mô hình) và chậm (hog + RF 296 s ở bước 2).
- **Val dự đoán test tốt**: tương quan hạng Spearman val–test 0.86 trên 16 cấu hình; chênh lệch test − val trung bình −0.002 (từ −0.035 đến +0.046). Riêng **hog + SVM** đứng thứ 4 trên val (0.445) nhưng cao nhất trên test (0.483). Không chọn lại theo test: chênh 0.035 trên val giữa sift + SVM và hog + SVM vẫn là căn cứ duy nhất được phép dùng.
- **Giá trị tốt nhất nằm ở biên lưới**: LogReg C = 0.01 (nhỏ nhất) ở hog, color, sift; SVM C = 10 (lớn nhất) ở hog; XGBoost 600 cây (lớn nhất) ở 3/4 đặc trưng. Lưới chưa đủ rộng (xem Cần làm trong README).

### Đánh giá từng lớp (test, `r03`)

Cấu hình chọn theo val (sift + SVM), so với hog + SVM (cao nhất trên test) và sift + LogReg (cấu hình chọn của `r02`). Đầy đủ 16 cấu hình trong notebook mục 4. P = precision, R = recall.

| Lớp (số ảnh test) | sift + SVM P / R / F1 | hog + SVM P / R / F1 | sift + LogReg P / R / F1 |
|---|---|---|---|
| hold (401) | 0.390 / 0.274 / 0.322 | **0.416 / 0.466 / 0.440** | 0.378 / 0.237 / 0.291 |
| ride (389) | 0.601 / 0.674 / 0.635 | **0.718 / 0.746 / 0.731** | 0.606 / 0.645 / 0.625 |
| eat/drink (208) | 0.295 / 0.361 / **0.325** | **0.300** / 0.308 / 0.304 | 0.284 / **0.370** / 0.322 |
| talk_on_phone (215) | 0.404 / **0.391** / **0.397** | **0.428** / 0.274 / 0.334 | 0.304 / 0.367 / 0.333 |
| work_on_computer (216) | 0.429 / **0.597** / **0.499** | **0.495** / 0.421 / 0.455 | 0.395 / 0.569 / 0.467 |
| unknown (457) | **0.642** / 0.569 / 0.603 | 0.604 / **0.659** / **0.630** | 0.625 / 0.519 / 0.567 |
| **macro** | 0.460 / 0.478 / 0.464 | **0.494 / 0.479 / 0.483** | 0.432 / 0.451 / 0.434 |
| accuracy | 0.488 | **0.526** | 0.457 |

Ma trận nhầm lẫn (số ảnh) của **sift + SVM** (hàng = nhãn thật, cột = dự đoán; `data/results/r03/confusion_sift_svm.csv`):

| Thật \ Dự đoán | hold | ride | eat/drink | talk_on_phone | work_on_computer | unknown |
|---|---|---|---|---|---|---|
| hold | **110** | 54 | 85 | 48 | 55 | 49 |
| ride | 23 | **262** | 14 | 19 | 17 | 54 |
| eat/drink | 36 | 20 | **75** | 27 | 43 | 7 |
| talk_on_phone | 41 | 19 | 29 | **84** | 20 | 22 |
| work_on_computer | 28 | 9 | 27 | 10 | **129** | 13 |
| unknown | 44 | 72 | 24 | 20 | 37 | **260** |

Nhận xét theo lớp:
- **Nhầm giữa 4 lớp "tay cầm vật" vẫn là lỗi chính**: 449/966 lỗi (46%) của sift + SVM, gần bằng `r02` (48%). Mô hình mạnh hơn không giải quyết được việc đặc trưng toàn ảnh không biết vật gì, ở đâu.
- **hog + SVM mạnh ở lớp dáng người**: `hold` recall 0.466 (sift + SVM 0.274), `ride` F1 0.731. HOG mã hoá dáng người (ngồi trên xe/ngựa, tay đưa ra cầm vật); SIFT-BoVW mạnh ở lớp có vật đặc trưng (`work_on_computer`, `talk_on_phone`). Hai đặc trưng bổ sung nhau, nên ghép hog + sift (hoặc late fusion 2 SVM) đáng thử (xem Cần làm trong README).
- **`ride` ↔ `unknown`** vẫn là cặp nhầm lớn thứ 2 (72 `unknown` → `ride`, 54 `ride` → `unknown`), như `r02`.

### Thời gian chạy `f02` (bước 1, 12.833 ảnh, 24/10/2026)

**Bước 1** (`f02`, 10 khối, `data/features/f02/timing_extract.csv`): tổng thời gian thực **247.2 s**.

| Công đoạn | Thời gian thực (s) |
|---|---|
| Lấy mẫu descriptor + KMeans K=500 (chỉ train) | 83.6 |
| Dense SIFT + BoVW (cả 3 split) | 130.5 |
| Đặc trưng ảnh hog/lbp/color/hog_upper/gist/skin (cả 3 split) | 32.2 |
| geom | 0.2 |

- CPU theo khối (cả 12.833 ảnh): sift_descriptor 842 s (65.6 ms/ảnh), bovw_encode 392 s, gist 136 s, lbp 82 s, hog 28 s (2.2 ms/ảnh).
- So với ước tính trước khi chạy (2–4 phút): thực tế 4 phút 7 s, vượt mức trên. Dense SIFT tốn ~66 ms/ảnh và bovw_encode ~31 ms/ảnh, gấp ~3.5 và ~11 lần `f01` (18.3 và 2.8 ms). **Nguyên nhân chưa xác định.** Không phải do kích thước crop: `sift_desc_pos` luôn resize về cạnh dài 256 px nên số điểm SIFT gần như không đổi. Giả thuyết (chưa kiểm chứng): `cpu_ms_per_img` đo bằng `perf_counter` trong từng tiến trình con, nên khi 10 tiến trình chạy lâu trên 4 nhân nhanh + 6 nhân chậm, cộng với `kmeans.predict` có thể tự mở thêm luồng trong mỗi tiến trình (oversubscription), thời gian mỗi ảnh bị đội lên; `f01` chỉ có 519 ảnh nên ít tranh chấp hơn.
- Các khối ảnh khác (lbp, color, gist, skin, hog_upper, geom) chỉ tốn ~25–30 s thời gian thực; KMeans và SIFT chiếm ~87% bước 1.

**Dung lượng `f02`** (đọc từ file; RAM bước 1 chưa đo vì `f02` trích trước khi có `resources.py`):
- Đặc trưng `f02` (cả 3 split, float32): tổng **534 MB**. Theo khối: hog 194.0, sift_spm 128.3, hog_upper 90.6, color 52.6, gist 26.3, sift 25.7, sift_upper 25.7, lbp 12.0, skin 1.1, geom 0.4 MB; codebook KMeans 1.1 MB. Dung lượng tỉ lệ với số chiều: hog (3780-d) gấp 7.5 lần sift (500-d).

### Thời gian chạy và tài nguyên `r03` (bước 2–4 trên `f02`, 24/10/2026)

Lệnh: `FEATS="hog lbp color sift" MODELS="logreg svm rf xgb" bash run_all.sh f02 r03 "..."` (dùng lại `f02`). Tổng **26 phút 56 s** (22:06:16 → 22:33:12, đo bằng giờ bắt đầu/kết thúc trong log).

| Bước | Thời gian thực | RAM đỉnh (MB) | RAM tăng thêm (MB) | CPU (s) | Số nhân dùng trung bình |
|---|---|---|---|---|---|
| 2. Chọn mô hình (16 cấu hình) | **1.330 s (22.2 phút)** | 7.837 | 7.688 | 8.072 | 6.1 |
| 3. Test (16 cấu hình, fit trên train+val) | 218 s (3.6 phút) | 2.378 | 2.203 | 910 | 4.2 |
| 4. Notebook | ~68 s (ước tính = tổng − bước 2 − bước 3) | – | – | – | – |

- **Bước 2 theo mô hình** (tổng 4 đặc trưng): RF 463 s, SVM 454 s, XGBoost 391 s, LogReg 22 s. Theo đặc trưng: color 539 s, hog 413 s, sift 224 s, lbp 154 s. Chậm nhất: hog + RF 296 s, color + SVM 237 s, color + XGB 215 s.
- **Bước 3**: fit lại 16 mô hình mất 202 s (SVM 79 s, XGB 68 s, RF 44 s, LogReg 11 s). Dự đoán chậm nhất: color + SVM 3.4 ms/ảnh (SVM giữ nhiều support vector); LogReg ≤ 0.02 ms/ảnh.
- **So với ước tính trước khi chạy** (17–25 phút cho cách test cũ, thêm 3–6 phút khi đổi sang test 16 cấu hình): thực tế 27 phút, ở mức trên.
  - Bước 2: 22.2 phút so với ước tính 15–21 phút. SVM bị ước tính thấp nhất: 7.6 phút so với 3.5–5.5 phút (color + SVM 237 s, sift + SVM 126 s; ước tính chỉ dựa trên 1 lần fit C = 1, trong khi C = 10 và 9 cấu hình chạy cùng lúc chậm hơn nhiều). hog + RF 296 s so với 237.5 s ở `r02` (cùng cấu hình).
  - XGBoost (6.5 phút) và LogReg khớp ước tính.
- **RAM**: đỉnh **7.8 GB** ở hog + RF, cao hơn nhiều so với ước tính "dưới 3 GB". Đây là cận trên (cộng RSS mọi worker, gồm nền ~10 worker loky và phần mảng memmap dùng chung bị đếm trùng); nguyên nhân chi tiết chưa tách. LogReg cũng có RAM đỉnh 4.5–6.2 GB vì phần lớn là nền của worker còn sống từ cấu hình trước. Máy có 25.8 GB nên chưa có vấn đề, nhưng bộ phối lớn (`fused_all` ~9.300 chiều) cần chú ý.
- **Dung lượng**: `data/results/r03/` 635 MB, trong đó mô hình 634 MB; 4 mô hình RF chiếm 426 MB, 4 SVM 173 MB, 4 XGB 34 MB, 4 LogReg 0.4 MB.

## Kết quả `r02` (bộ gộp 4 nguồn, unknown 3000, 24/10/2026)

Chạy trên **12.833 ảnh** (`data/train_test_split/labels_split_unk3000.csv`: train 9.069, val 1.878, test 1.886), 2 đặc trưng (`sift`, `hog`) × 2 mô hình (LogReg, RF). Biểu đồ đầy đủ trong `code/04_report.ipynb`.
Test có `unknown` đã giảm (23% thay vì 56%), nên precision các lớp hành động cao hơn so với phân bố thật; macro-F1 vẫn so được giữa các cấu hình trong `r02`, nhưng **không so trực tiếp được với `r01`** (khác dữ liệu, khác test).

| Đặc trưng | Mô hình | Siêu tham số tốt nhất | Val macro-F1 | Test macro-F1 | Test balanced acc (= macro recall) | Test macro precision | Test accuracy (tham khảo) |
|---|---|---|---|---|---|---|---|
| sift (500-d) | LogReg | C = 0.01 | **0.430** | **0.434** | **0.451** | 0.432 | 0.457 |
| sift (500-d) | RF | 300 cây, depth None, leaf 3 | 0.415 | 0.432 | 0.437 | **0.455** | **0.468** |
| hog (3780-d) | LogReg | C = 0.01 | 0.362 | 0.387 | 0.392 | 0.386 | 0.411 |
| hog (3780-d) | RF | 300 cây, depth None, leaf 3 | 0.339 | – | – | – | – |

Test macro precision và accuracy tính từ `test_pred_*.npy` (cùng số với `test_report.txt`). Accuracy của sift + RF cao hơn LogReg dù macro-F1 thấp hơn, vì RF đoán đúng nhiều hơn ở lớp lớn (`unknown`, `ride`): đúng lý do không dùng accuracy để chọn mô hình.

hog + RF không chạy test vì không phải cấu hình tốt nhất của đặc trưng `hog` hay của mô hình RF (`03_final_test.py` chỉ test các cấu hình đó).

Recall từng lớp trên test:

| Cấu hình | hold | ride | eat/drink | talk_on_phone | work_on_computer | unknown |
|---|---|---|---|---|---|---|
| sift + LogReg | 0.237 | 0.645 | **0.370** | **0.367** | **0.569** | 0.519 |
| sift + RF | **0.322** | **0.666** | 0.197 | 0.293 | 0.551 | **0.595** |
| hog + LogReg | 0.287 | 0.563 | 0.284 | 0.321 | 0.403 | 0.495 |

Nhận xét:
- **Val khớp test**: chênh lệch val–test ≤ 0.03 ở cả 2 cấu hình (với `r01`, val 42 ảnh, có chênh tới 0.18). Val 1.878 ảnh đủ để chọn mô hình.
- **sift tốt hơn hog ở 5/6 lớp**; hog chỉ hơn ở `hold` (0.287 so với 0.237).
- `hold` là lớp khó nhất (recall 0.24–0.29): "cầm vật" trông giống `unknown` và các lớp có cầm vật khác (`eat/drink`, `talk_on_phone`). `ride` dễ nhất (0.56–0.65).
- C tốt nhất là **0.01, giá trị nhỏ nhất của lưới** ở cả 2 đặc trưng: có thể C nhỏ hơn còn tốt hơn (xem Cần làm trong README).
- **RF ngang LogReg về macro-F1 trên sift** (test 0.432 so với 0.434) nhưng thiên về lớp lớn: recall `hold`, `ride`, `unknown` cao hơn, còn `eat/drink` (0.197) và `talk_on_phone` (0.293) thấp hơn hẳn, dù đã dùng `class_weight="balanced_subsample"`. Trên hog, RF lệch mạnh hơn nữa (val: recall `talk_on_phone` 0.073, `eat/drink` 0.110).
- RF tốn hơn nhiều: hog + RF mất 237.5 s ở bước 2, so với 14.6 s của hog + LogReg.

### Đánh giá từng lớp (test, `r02`)

Tính từ `data/results/r02/test_pred_*.npy` và `data/features/f02/y_test.npy` (cùng số với `test_report.txt` và `confusion_*.png`). P = precision, R = recall.

| Lớp (số ảnh test) | sift + LogReg P / R / F1 | sift + RF P / R / F1 | hog + LogReg P / R / F1 |
|---|---|---|---|
| hold (401) | 0.378 / 0.237 / 0.291 | 0.362 / **0.322** / **0.341** | 0.346 / 0.287 / 0.314 |
| ride (389) | **0.606** / 0.645 / **0.625** | 0.515 / **0.666** / 0.581 | 0.578 / 0.563 / 0.570 |
| eat/drink (208) | 0.284 / **0.370** / **0.322** | 0.281 / 0.197 / 0.232 | 0.233 / 0.284 / 0.256 |
| talk_on_phone (215) | 0.304 / **0.367** / 0.333 | **0.600** / 0.293 / **0.394** | 0.265 / 0.321 / 0.291 |
| work_on_computer (216) | 0.395 / **0.569** / 0.467 | **0.408** / 0.551 / **0.469** | 0.345 / 0.403 / 0.372 |
| unknown (457) | **0.625** / 0.519 / 0.567 | 0.562 / **0.595** / **0.578** | 0.551 / 0.495 / 0.521 |
| **macro** | **0.432 / 0.451 / 0.434** | 0.455 / 0.437 / 0.432 | 0.386 / 0.392 / 0.387 |

Ma trận nhầm lẫn của **sift + LogReg** (cấu hình tốt nhất trên val; hàng = nhãn thật, cột = dự đoán):

| Thật \ Dự đoán | hold | ride | eat/drink | talk_on_phone | work_on_computer | unknown |
|---|---|---|---|---|---|---|
| hold | **95** | 45 | 93 | 66 | 60 | 42 |
| ride | 23 | **251** | 22 | 27 | 19 | 47 |
| eat/drink | 26 | 14 | **77** | 31 | 49 | 11 |
| talk_on_phone | 40 | 15 | 36 | **79** | 23 | 22 |
| work_on_computer | 18 | 10 | 26 | 19 | **123** | 20 |
| unknown | 49 | 79 | 17 | 38 | 37 | **237** |

Nhận xét theo lớp:
- **Phần lớn lỗi là nhầm giữa 4 lớp "tay cầm vật"** (`hold`, `eat/drink`, `talk_on_phone`, `work_on_computer`): 487/1.024 lỗi (48%) của sift + LogReg nằm trong nhóm này. Đặc trưng toàn ảnh (BoVW không giữ vị trí, HOG trên cả crop) thấy "có tay, có vật nhỏ" nhưng không phân biệt được vật gì và vật ở đâu so với đầu/tay.
- **`hold` khó nhất** (F1 0.29–0.34) và bị nhầm **ra mọi phía**: với sift + LogReg, 93 ảnh (23%) thành `eat/drink`, 66 (16%) thành `talk_on_phone`, 60 (15%) thành `work_on_computer`. Theo định nghĩa lớp, `hold` là "cầm vật nhưng không làm 3 hành động kia", nên về hình ảnh nó gần như bao trùm 3 lớp đó. Vì `hold` bị dự đoán thành các lớp này nhiều, precision của `eat/drink` (0.28) và `talk_on_phone` (0.30) thấp.
- **`eat/drink` hay bị nhầm thành `work_on_computer`** (49/208 = 24%). Giả thuyết (chưa kiểm chứng bằng ảnh): ăn uống thường ở bàn, cùng bối cảnh với ngồi trước máy tính.
- **`ride` ↔ `unknown`** là cặp nhầm lớn thứ 2: 79 ảnh `unknown` (17%) bị đoán là `ride`, 47 ảnh `ride` bị đoán là `unknown`. Giả thuyết (chưa kiểm chứng): người đứng cạnh xe đạp, ngựa, hoặc ở cảnh đường phố có bối cảnh giống `ride`.
- **`work_on_computer` tốt thứ 2** sau `ride` (recall 0.57): laptop/bàn phím có kết cấu đặc trưng, SIFT bắt tốt.
- **RF và LogReg đánh đổi precision/recall khác nhau**: RF có precision `talk_on_phone` cao (0.60) nhưng recall thấp (0.29), tức RF chỉ đoán `talk_on_phone` khi rất chắc. Ma trận nhầm lẫn của RF có nhiều ảnh bị đẩy về `ride` và `unknown` hơn (ví dụ `hold` → `ride` 83, `hold` → `unknown` 71), khớp với nhận xét RF thiên về lớp lớn.

### Thời gian chạy và tài nguyên `r02` (bước 2–4 trên `f02`, 24/10/2026)

Thời gian bước 1 (`f02`) xem mục Thời gian chạy của README.

**Bước 2** (`r02`, 2 cấu hình × 4 giá trị C): tổng **16.3 s**. sift + LogReg: grid 1.5 s, fit lại 0.13 s; hog + LogReg: grid 11.6 s, fit lại 2.9 s.

**Bước 3** (2 cấu hình): tổng **4.4 s**. Fit trên train+val: sift 0.15 s, hog 3.9 s; dự đoán ≤ 0.03 ms/ảnh.

**Bước 2–4 bằng `run_all.sh`** (dùng lại `f02`, đo bằng `time`): **1 phút 14 s**; notebook khoảng 53 s (ước tính = 74 s trừ bước 2–3, gồm cả RF tính importance trên sift + hog).

**Bổ sung RF vào `r02`** (`02_train_select.py --update --feats sift hog --models rf`, rồi `03_final_test.py --update` và notebook; đo bằng `time`): **5 phút 24 s**.
- Bước 2: **262.1 s**. sift + RF: grid 22.8 s + fit lại 1.7 s; hog + RF: grid **216.5 s** + fit lại 20.9 s.
- Bước 3 (3 cấu hình: sift + LogReg, hog + LogReg, sift + RF): 7.0 s; fit sift + RF trên train+val 2.3 s.
- Notebook khoảng 55 s (ước tính = 324 s trừ bước 2–3).
- So với ước tính trước khi chạy (5–7 phút; hog + RF grid 2.5–4.5 phút): khớp. Ước tính dựa trên đo 20 cây trên 1 nhân (sift ~0.03 s/cây, hog ~0.28 s/cây).

**Tài nguyên `r02`** (đọc từ file trên đĩa; RAM chưa đo vì `r02` chạy trước khi có `resources.py`):
- Mô hình `r02` (`models/*.joblib`): sift + LogReg **0.04 MB**, hog + LogReg 0.27 MB, sift + RF **114.6 MB**. RF lớn gấp ~3.000 lần LogReg trên cùng đặc trưng (300 cây, `max_depth=None`) mà macro-F1 không hơn; `data/results/r02/` (110 MB) gần như toàn bộ là file RF này.

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
- **Ước tính thời gian SVM từ 1 lần fit thường thấp hơn thực tế**: ở `r03`, ước tính 3.5–5.5 phút từ 1 lần fit với C = 1, thực tế 7.6 phút. C lớn (10) hội tụ chậm hơn nhiều, và 9 cấu hình chạy cùng lúc trên 4 nhân nhanh + 6 nhân chậm làm mỗi fit chậm thêm. Nên đo cả cấu hình chậm nhất của lưới (C lớn nhất) trước khi ước tính.
- **RAM đỉnh khi grid search song song gấp nhiều lần kích thước dữ liệu**: `r03` đạt 7.8 GB (hog + RF) trong khi ma trận hog train+val chỉ 164 MB. Khi ước tính RAM phải tính nền của ~10 worker (mỗi worker đã nạp numpy/sklearn) và số mô hình fit đồng thời (8 rừng 300 cây không giới hạn độ sâu), không chỉ kích thước X.
- **Chênh lệch nhỏ trên val không đáng tin để xếp hạng**: ở `r03`, sift + SVM hơn hog + SVM 0.035 trên val nhưng thua 0.019 trên test. Dù val tương quan tốt với test (Spearman 0.86), các cấu hình cách nhau ~0.03 nên coi là ngang nhau; muốn phân định thì cần cross-validation hoặc nhiều seed.
- **Mô hình tuyến tính đánh giá thấp đặc trưng histogram**: `color` với LogReg kém nhất (0.335) nhưng với XGBoost đứng thứ 2 (0.458). Không nên loại một đặc trưng chỉ vì kết quả với LogReg.
- **Zero-shot cần prompt cho mọi lớp, kể cả lớp "còn lại"**: ở `r04`, `unknown` không có prompt nên softmax giữa 5 prompt hành động luôn chọn 1 lớp; 65% ảnh `unknown` bị đoán `ride` với xác suất cao, ngưỡng T không lọc được. Ngưỡng trên xác suất chỉ bắt được ảnh "mơ hồ", không bắt được ảnh "tự tin nhưng sai lớp".
- **Lớp định nghĩa bằng phủ định khó diễn đạt bằng văn bản**: `hold` ("cầm vật nhưng không ăn/uống, gọi điện, dùng máy tính") có recall zero-shot 0.10 nhưng linear probe 0.85. Với lớp kiểu này, cần nhãn giám sát.
- **Ngưỡng T của softmax CLIP phải đặt trên xác suất, không trên cosine**: cosine ảnh–văn bản của CLIP chỉ khoảng 0.2–0.3, nên ngưỡng 0.7/0.8 chỉ có nghĩa với xác suất sau softmax (logit = 100 × cosine).
- **Ước tính tốc độ từ lần chạy thử rất nhỏ bị lệch do chi phí khởi động**: chạy thử CLIP 60 ảnh cho ~60 ảnh/giây (gồm nạp mô hình), thực tế 85 ảnh/giây; ước tính CLIP 3.5–5 phút, thực tế 2.5 phút.

# Thực nghiệm

Nhật ký đầy đủ các thay đổi đặc trưng / mô hình nhằm cải thiện kết quả, kèm lý do và số đo. `README.md` chỉ giữ thực nghiệm mới nhất; phiên bản `fXX` / `rXX` tra trong mục Phiên bản kết quả của [README](README.md#phiên-bản-kết-quả).

Mục 1–4 chạy trên mẫu **519 ảnh** V-COCO (train 261, val 42, test 216); mục 5 trở đi chạy trên bộ gộp 4 nguồn (12,833 ảnh, train 9,069 / val 1,878 / test 1,886). Thời gian đo trên Apple Silicon, 10 CPU.

**Quy ước tên** (`config.FEATURE_NAMES`, `config.MODEL_NAMES`): bảng, hình, nhận xét và slide dùng **tên hiển thị**; tên trong code chỉ xuất hiện ở tên file, lệnh chạy và đoạn code. Không đổi khoá trong code vì mọi file đặc trưng / kết quả (`f01`–`f03`, `r01`–`r05`) đặt theo khoá này.

| Tên hiển thị | Tên trong code | Tên đầy đủ | Cách tính (đúng như `code/features.py`) | Số chiều |
|---|---|---|---|---|
| HOG | `hog` | Histogram of Oriented Gradients | ảnh xám resize 64×128; 9 hướng, ô 8×8, khối 2×2, chuẩn hoá L2-Hys (skimage) | 3,780 |
| LBP | `lbp` | LBP histogram (uniform, bất biến xoay) | ảnh xám resize 128×256; LBP `uniform` P = 24, R = 3 (skimage: 26 mẫu uniform bất biến xoay); histogram 26 bin trên mỗi ô của lưới 3×3, chuẩn hoá L1 từng ô | 234 |
| Color-HSV | `color` | Color histogram HSV | ảnh gốc (không resize) đổi sang HSV; histogram 3 chiều 8×8×8 cho nửa trên và nửa dưới, chuẩn hoá L1 | 1,024 |
| SIFT-BoVW | `sift` | Dense SIFT + Bag of Visual Words | ① dense SIFT (OpenCV) trên ảnh xám resize cạnh dài 256, lưới bước 8, cỡ 16: mỗi điểm 1 descriptor 128 chiều; ② codebook MiniBatchKMeans K = 500 học trên ~200,000 descriptor lấy mẫu từ train; ③ gán mỗi descriptor vào từ gần nhất, đếm thành histogram, chuẩn hoá L1 rồi căn bậc 2 (Hellinger) | 500 |

Mốc so sánh: **CLIP** (`clip`, embedding ảnh CLIP ViT-B/32, 512 chiều). Mô hình: **LogReg** (`logreg`), **SVM** (`svm`, kernel RBF), **RF** (`rf`), **XGBoost** (`xgb`), **zero-shot** (`clip_zs`). Cấu hình viết là "đặc trưng + mô hình", ví dụ SIFT-BoVW + SVM.

1. **Bản đầu: 4 đặc trưng cơ bản × 3 mô hình** (12/10/2026)
   - Vấn đề / lý do: yêu cầu đề bài (3 loại đặc trưng cơ bản, 3 mô hình ML cơ bản); người làm đồ án chọn 4 đặc trưng.
   - Thay đổi:
     - Đặc trưng: HOG (64×128, 3,780-d), LBP (P=24, R=3, uniform, lưới 3×3 để giữ vị trí, 234-d), Color-HSV (8×8×8, nửa trên/dưới, 1,024-d), SIFT-BoVW (dense SIFT bước 8 px, size 16; codebook MiniBatchKMeans K=500 chỉ học trên train; chuẩn hóa Hellinger; 500-d), cộng `fused` = nối 4 khối.
     - Mô hình: LogReg, SVM RBF (PCA 256 cho HOG/`fused` để giảm chiều), RF.
     - Cân bằng lớp bằng `class_weight` (không oversample, để giữ phân bố thật). Chọn siêu tham số bằng `GridSearchCV` + `PredefinedSplit` (train → val), scoring macro-F1.
   - Kết quả (mốc ban đầu):
     - Val tốt nhất: SIFT-BoVW + SVM 34.5.
     - Test tốt nhất: Fused + LogReg 25.4.
     - Thời gian: chưa đo.
   - Nhận xét: RF thường chỉ đoán `unknown` khi mỗi lớp chỉ có vài chục ảnh.
   - Phiên bản: không còn file (xem bảng [Phiên bản kết quả](README.md#phiên-bản-kết-quả)).
2. **Thêm XGBoost** (12/10/2026)
   - Vấn đề / lý do: người làm đồ án yêu cầu; XGBoost thường mạnh với dữ liệu dạng bảng.
   - Thay đổi:
     - Cân bằng lớp bằng `sample_weight`, vì XGBoost multi-class không có `class_weight`.
     - Lưới: `n_estimators` {300, 600}, `max_depth` {4, 6}, `learning_rate` {0.05, 0.1}; cố định `subsample` 0.8, `colsample_bytree` 0.5, `max_bin` 256 (mặc định).
   - Kết quả:
     - Val tốt nhất không đổi: 34.5 → 34.5 (SIFT-BoVW + SVM). Tốt nhất của XGBoost: SIFT-BoVW + XGBoost 23.9; các bộ khác 15–21.
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
     - PCA cho SVM chuyển từ danh sách cố định sang tự động khi số chiều > 1,500.
     - Cách đánh giá: `03_final_test.py` chạy test thêm cho mô hình tốt nhất của từng loại mô hình.
   - Kết quả:
     - Val tốt nhất không đổi: 34.5 → 34.5 (SIFT-BoVW + SVM). Đặc trưng mới tốt nhất trên val: `fused_all` + LogReg 32.3, `upper` + SVM 31.9.
     - Test tốt nhất: 25.4 → **26.7** (`geom` + RF). Ngoài ra `fused_all` + LogReg 24.8 có balanced accuracy cao nhất (30.6); `upper` + SVM 24.4.
     - Thời gian bước 2 (48 cấu hình): 8 phút 19 s, đo bằng `time`; XGBoost chiếm 461.9 s.
   - Nhận xét: `geom` (8 chiều) có importance trung bình mỗi chiều cao nhất trong RF trên `fused_all`, tiếp theo là `skin`.
   - Phiên bản: không còn file; đặc trưng cùng tham số với `f01`.
4. **Sửa XGBoost chạy chậm** (12/10/2026)
   - Vấn đề / lý do:
     - XGBoost chiếm 461.9/499 s của bước 2, dù chỉ có 261 ảnh train.
     - Đo riêng 1 lần fit trên `fused_all` (~9,300 chiều, 300 cây): 28.3 s.
     - Với đủ dữ liệu, riêng XGBoost có thể chạy hàng giờ (ước tính, chưa đo).
   - Thay đổi (`code/models.py`):
     - `max_bin` 256 → 64.
     - Thêm `PCA(256)` trước XGBoost khi bộ đặc trưng > 1,500 chiều (áp dụng cho `hog`, `sift_spm`, `upper`, `fused`, `combo_local`, `fused_all`).
   - Kết quả:
     - Thời gian: XGBoost cả 12 bộ đặc trưng 461.9 s → 46 s; riêng `max_bin=64` (chưa PCA) giảm 1 lần fit từ 28.3 s xuống 15.0 s.
     - Val tốt nhất: 34.5 → **35.2** (SIFT-BoVW + XGBoost, trước đó 23.9). `geom` + XGBoost 26.6 vượt `geom` + RF 25.5.
     - Test tốt nhất: 26.7 → **28.3** (`geom` + XGBoost, thay cho `geom` + RF). Nhưng cấu hình tốt nhất trên val (SIFT-BoVW + XGBoost) chỉ đạt **17.3** trên test.
   - Nhận xét:
     - Bộ SIFT-BoVW chỉ có 500 chiều nên **không bị PCA**: mức tăng của SIFT-BoVW + XGBoost đến từ `max_bin=64`, cộng thêm độ nhiễu của val 42 ảnh, không phải từ PCA.
     - Val nhỏ dễ chọn nhầm: tốt nhất trên val lại gần mức đoán ngẫu nhiên trên test.
   - Phiên bản: kết quả cuối được tái lập trong `r01` (bước 2 mất 69.1 s).

5. **Chuyển sang bộ gộp 4 nguồn, giảm `unknown`, chỉ giữ SIFT-BoVW + HOG × LogReg** (24/10/2026)
   - Vấn đề / lý do:
     - `r01` chỉ có 519 ảnh (val 42): chọn mô hình rất nhiễu (val tốt nhất 35.2 nhưng test 17.3).
     - Bộ gộp `dataset_6class_crops` có 22,366 ảnh, nhiều ảnh hành động hơn hẳn bộ V-COCO đang có.
     - `unknown` chiếm 56%; giảm xuống 3,000 để các lớp cân bằng hơn và chạy nhanh hơn.
     - Chạy đủ 12 bộ × 4 mô hình ước tính 1–1.5 giờ; người làm đồ án chọn chỉ giữ 2 đặc trưng cơ bản bổ sung cho nhau (SIFT-BoVW: chi tiết vật; HOG: dáng người) và 1 mô hình (LogReg, ổn định nhất ở `r01`). `geom` bị loại vì dựa vào bbox trong dữ liệu chứ không phải đặc trưng ảnh truyền thống.
   - Thay đổi:
     - Dữ liệu: `vcoco_6class_crops/labels.csv` → `train_test_split/labels_split_unk3000.csv` (chia 70/15/15 theo ảnh gốc, phân tầng lớp × nguồn, seed 2026; giữ 3,000 `unknown` đầu sau khi xáo).
     - Đặc trưng dùng ở bước 2: 12 bộ → SIFT-BoVW, HOG. Mô hình: 4 → LogReg (lưới C {0.01, 0.1, 1, 10} không đổi). Tham số đặc trưng không đổi.
   - Kết quả (`f02` / `r02`, 12,833 ảnh):
     - SIFT-BoVW + LogReg: val 33.1 → **43.0**, test 26.2 → **43.4** (`r01` → `r02`).
     - HOG + LogReg: val 21.9 → 36.2; test 38.7 (`r01` không chạy test cho HOG + LogReg).
     - Chênh lệch val–test ≤ 3 (ở `r01` tới 18).
     - Thời gian: bước 1 247 s, bước 2 16.3 s, bước 3 4.4 s.
   - Nhận xét / tác dụng phụ:
     - **Không so trực tiếp với `r01`**: khác dữ liệu, khác test, và test của `r02` đã giảm `unknown`. Phần tăng chủ yếu đến từ có nhiều dữ liệu hơn (train 261 → 9,069) và phân bố lớp cân bằng hơn, không phải từ thay đổi mô hình.
     - C tốt nhất là 0.01 (biên dưới của lưới) cho cả 2 đặc trưng, nên lưới C chưa đủ rộng.
     - Bỏ các bộ phối và `geom` nên không còn mốc "phối có hơn đơn lẻ không" trên dữ liệu lớn.
   - Phiên bản: `f02`, `r02` (lần chạy dở dang: `f02_interrupted`, `r02_interrupted`, không dùng).

6. **Thêm Random Forest vào `r02`** (24/10/2026)
   - Vấn đề / lý do: người làm đồ án muốn so LogReg với một mô hình phi tuyến trên cùng đặc trưng SIFT-BoVW, HOG của `f02`. Ước tính trước khi chạy: 5–7 phút, chủ yếu là HOG + RF.
   - Thay đổi: bổ sung RF vào `r02` bằng `--update` (người làm đồ án đồng ý). Lưới RF giữ nguyên như `r01`: `n_estimators` {300, 500}, `max_depth` {None, 30}, `min_samples_leaf` {1, 3}, `class_weight="balanced_subsample"`.
   - Kết quả:
     - SIFT-BoVW + RF: val 41.5, test **43.2** (SIFT-BoVW + LogReg: val 43.0, test 43.4).
     - HOG + RF: val 33.9 (HOG + LogReg: 36.2); không chạy test.
     - Cả 2 đặc trưng chọn cùng cấu hình: 300 cây, `max_depth=None`, `min_samples_leaf=3`.
     - Thời gian bước 2: SIFT-BoVW + RF 24.5 s, HOG + RF 237.5 s (LogReg: 1.7 s và 14.6 s).
   - Nhận xét / tác dụng phụ:
     - RF **không hơn** LogReg về macro-F1, mà tốn gấp 14–16 lần thời gian ở bước 2.
     - RF đổi kiểu lỗi: tăng recall `hold` (23.7 → 32.2), `ride`, `unknown`, nhưng giảm mạnh `eat/drink` (37.0 → 19.7) và `talk_on_phone` (36.7 → 29.3) trên test. Hai mô hình sai ở các lớp khác nhau, nên late fusion LogReg + RF có thể có ích (chưa thử).
     - `min_samples_leaf=3` thắng 1 ở cả 2 đặc trưng: lá lớn hơn (ít khớp quá mức) tốt hơn.
   - Phiên bản: `r02` (cập nhật bằng `--update`; `run_info.json` ghi cả 2 lần chạy bước 2 và 3).

7. **4 đặc trưng cơ bản × 4 mô hình, đủ chỉ số và đo RAM** (24/10/2026)
   - Vấn đề / lý do: `r02` mới thử 2 đặc trưng (SIFT-BoVW, HOG) × 2 mô hình (LogReg, RF) trên dữ liệu đầy đủ; chưa biết LBP, Color-HSV và SVM, XGBoost làm được gì khi có đủ dữ liệu (kết quả trên mẫu 519 ảnh không đáng tin). Đồng thời đánh giá chưa đủ chỉ số (thiếu accuracy, precision, confusion số ảnh) và chưa đo RAM.
   - Thay đổi:
     - Chạy HOG, LBP, Color-HSV, SIFT-BoVW × LogReg, SVM, RF, XGBoost trên `f02`. Lưới giữ nguyên như `r01`.
     - Cách đánh giá: `eval_metrics` (models.py) ghi accuracy, macro precision, weighted F1, P/R/F1 từng lớp cho cả val và test; confusion matrix dạng số ảnh (csv + png); lưu `val_pred_*.npy`.
     - Bước 3 **test mọi cấu hình** (trước đây chỉ test cấu hình tốt nhất mỗi đặc trưng / mỗi mô hình); cấu hình chọn vẫn theo val, riêng cho từng nhóm (`group`).
     - Đo RAM đỉnh, CPU, dung lượng mô hình (`code/resources.py`).
   - Kết quả:
     - Cấu hình chọn theo val: **SIFT-BoVW + SVM** (C = 1), val 48.0, test **46.4** (`r02`: SIFT-BoVW + LogReg, test 43.4; +3.0).
     - SVM tốt nhất trên val cho SIFT-BoVW, HOG, LBP; XGBoost cho Color-HSV (Color-HSV + XGBoost val 45.8, Color-HSV + LogReg chỉ 33.5).
     - HOG + SVM: val 44.5 (thứ 4) nhưng test cao nhất 48.3; recall `hold` 46.6 so với 27.4 của SIFT-BoVW + SVM.
     - Tổng thời gian 26 phút 56 s (ước tính 17–25 phút + 3–6 phút do test 16 cấu hình); RAM đỉnh bước 2: 7.8 GB (HOG + RF).
   - Nhận xét / tác dụng phụ:
     - Cải thiện đến từ **mô hình phi tuyến** (SVM RBF), không phải đặc trưng mới: cùng SIFT-BoVW, LogReg 43.4 → SVM 46.4 trên test.
     - Lỗi chính vẫn là nhầm giữa 4 lớp "tay cầm vật" (46% lỗi, `r02` 48%).
     - Val đáng tin (Spearman val–test 0.86 trên 16 cấu hình), nên việc chọn theo val ổn; nhưng 3.5 chênh lệch giữa SIFT-BoVW + SVM và HOG + SVM trên val đảo chiều trên test, nên các cấu hình trong khoảng ~3 nên coi là ngang nhau.
     - RF không đáng dùng tiếp: kém nhất trong 3 mô hình phi tuyến, mô hình 98–115 MB.
     - Nhiều siêu tham số tốt nhất nằm ở biên lưới (LogReg C nhỏ nhất, SVM C lớn nhất ở HOG, XGBoost 600 cây).
   - Phiên bản: `r03` (đặc trưng `f02`).

8. **CLIP ViT-B/32: zero-shot với ngưỡng T và bộ phân loại có giám sát (mốc so sánh)** (24/10/2026)
   - Vấn đề / lý do: ở `r03`, 46% lỗi của cấu hình tốt nhất là nhầm giữa 4 lớp "tay cầm vật": đặc trưng thủ công thấy có tay, có vật nhỏ nhưng không biết vật gì. CLIP mã hoá ngữ nghĩa vật / hành động. Người làm đồ án đề xuất zero-shot theo ý tưởng ngưỡng T chọn Reliable Person Query của DP-HOI, với T tham khảo 0.7 và 0.8.
   - Thay đổi:
     - Khối đặc trưng CLIP (`code/clip_feat.py`, `01_extract_features.py --clip`): embedding ảnh CLIP ViT-B/32 `openai`, 512 chiều, chuẩn hoá L2; ảnh crop được **đệm thành hình vuông** (tiền xử lý mặc định cắt giữa sẽ mất đầu/chân).
     - Mô hình `clip_zs` (`models.ClipZeroShot`): 5 prompt "a photo of a person {holding something / riding / eating or drinking / talking on the phone / using a computer}"; xác suất = softmax(100 × cosine); lớp cao nhất, nếu xác suất cao nhất < T thì `unknown`. **T đặt trên xác suất softmax** (cosine thô của CLIP chỉ ~0.2–0.3 nên T = 0.7/0.8 không có nghĩa với cosine). Lưới T ∈ {0, 0.5, 0.6, 0.7, 0.8, 0.9}, chọn trên val.
     - Bộ phân loại có giám sát trên embedding đóng băng: CLIP × LogReg (linear probe đúng nghĩa), SVM RBF, RF, XGBoost (phi tuyến) như các đặc trưng khác. (Sửa thuật ngữ 31/10/2026: trước gọi chung cả 4 là "linear probe", sai với SVM RBF, RF, XGBoost.) Kết quả CLIP thuộc nhóm CLIP, chọn tốt nhất riêng với nhóm thủ công.
     - Lưu điểm val của từng giá trị trong lưới (`grid_*.csv`).
   - Kết quả (`f03`, `r04`):
     - Zero-shot: val chọn **T = 0.6** (60.5); T = 0.7: 60.3, T = 0.8: 59.9, không ngưỡng: 56.2. Test **58.1** (accuracy 54.7).
     - Có giám sát: **CLIP + SVM RBF** val 87.8, test **88.9** (accuracy 89.2); linear probe (CLIP + LogReg) test 87.6.
     - So với đặc trưng thủ công tốt nhất theo val (SIFT-BoVW + SVM, test 46.4): zero-shot +11.7, linear probe (LogReg) +41.2, SVM RBF +42.5.
     - Thời gian: trích CLIP 151.7 s cho 12,833 ảnh trên CPU; cả pipeline 15 phút 37 s.
   - Nhận xét / tác dụng phụ:
     - **Ngưỡng T có ích nhưng không nhạy**: có ngưỡng tăng val 56.2 → 60.5; mọi T từ 0.5 đến 0.9 cho 59.5–60.5. Hai giá trị tham khảo 0.7 và 0.8 không hơn 0.6.
     - **Zero-shot hỏng ở 2 lớp định nghĩa bằng phủ định**: 65% ảnh `unknown` bị đoán là `ride` (không có prompt cho `unknown`, softmax luôn chọn 1 trong 5 lớp); 58% ảnh `hold` bị gán `unknown` ("holding something" quá mơ hồ). Lớp có vật rõ thì tốt (`work_on_computer` F1 89.9).
     - Bộ phân loại có giám sát giải quyết đúng 2 lớp đó (recall `hold` / `unknown`: LogReg 80.3 / 86.9, SVM RBF 84.8 / 90.2): cần nhãn để học ranh giới "cầm vật nhưng không làm 3 hành động kia".
     - So sánh với DP-HOI chỉ là mô phỏng lỏng: ở bài báo T = 0.9 là ngưỡng điểm phát hiện người để chọn RPQ, ở đây T là ngưỡng tin cậy hành động.
     - **Rủi ro dữ liệu**: CLIP train trên dữ liệu web không công bố, có thể chứa ảnh COCO (gồm ảnh test V-COCO). Kết quả CLIP chỉ là mốc so sánh, không thay kết quả chính của đặc trưng thủ công.
   - Phiên bản: `f03` (f02 + khối CLIP), `r04`.

Kết luận tạm: trên mẫu nhỏ, các tổ hợp (`fused`, `fused_all`, `combo_local`) ổn định hơn đặc trưng đơn lẻ; `geom` là đặc trưng rẻ mà hiệu quả nhất; LogReg ổn nhất với dữ liệu ít và nhiều chiều. **Chưa có kết luận nào đáng tin cho tới khi chạy đủ dữ liệu.**

## Kết quả `r04` (CLIP so với SIFT-BoVW, 24/10/2026)

Chạy trên **12,833 ảnh** (`f03` = các khối của `f02` + khối CLIP; khối thủ công của `f03` giống từng bit `f02`), 2 bộ đặc trưng × mô hình, **test cả 9 cấu hình**:
- **Nhóm CLIP** (mốc so sánh, không phải đặc trưng thủ công): embedding ảnh CLIP ViT-B/32 (512 chiều) × LogReg (linear probe), SVM RBF, RF, XGBoost (phi tuyến) và `clip_zs` (zero-shot).
- **Nhóm thủ công**: SIFT-BoVW × 4 mô hình, đặt cạnh để so (số liệu trùng `r03`).

> **Lưu ý dữ liệu**: CLIP được train trên ~400 triệu cặp ảnh–văn bản lấy từ web, không công bố. Không loại trừ được việc dữ liệu đó có ảnh COCO, kể cả ảnh test V-COCO (trong test của `r04`, 1,025/1,886 ảnh là từ v_coco, tức ảnh COCO). Vì vậy kết quả CLIP chỉ là **mốc so sánh**, không thay kết quả chính của đặc trưng thủ công.

Xếp theo nhóm, rồi theo val macro-F1. Cấu hình chọn theo val trong từng nhóm: **CLIP + SVM** và **SIFT-BoVW + SVM**. Accuracy chỉ để tham khảo; macro recall = balanced accuracy. RAM là RAM đỉnh khi grid search ở bước 2 (cận trên, gồm nền worker).

| Nhóm | Đặc trưng | Mô hình | Siêu tham số tốt nhất | Val macro-F1 | Test macro-F1 | Test accuracy | Test macro precision | Test macro recall | Test weighted F1 | Bước 2 (s) | RAM bước 2 (MB) | Mô hình (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP | **CLIP** | **SVM** | C = 10, γ = scale | **87.8** | **88.9** | **89.2** | **89.3** | **88.5** | **89.2** | 46.9 | 3,942 | 23.90 |
| CLIP | CLIP | XGBoost | lr 0.1, depth 4, 600 cây | 87.0 | 88.2 | 88.5 | 88.7 | 87.9 | 88.6 | 102.1 | 4,864 | 5.68 |
| CLIP | CLIP | LogReg | C = 0.01 | 87.0 | 87.6 | 88.0 | 87.2 | 88.1 | 88.0 | 3.5 | 1,441 | 0.04 |
| CLIP | CLIP | RF | 300 cây, depth None, leaf 3 | 84.7 | 83.3 | 83.6 | 86.5 | 82.0 | 83.9 | 99.0 | 4,295 | 69.21 |
| CLIP | CLIP | **zero-shot** | T = 0.6 | 60.5 | 58.1 | 54.7 | 65.1 | 61.5 | 50.8 | 0.3 | 4,255 | 0.01 |
| thủ công | **SIFT-BoVW** | **SVM** | C = 1, γ = scale | 48.0 | 46.4 | 48.8 | 46.0 | 47.8 | 48.4 | 109.5 | 4,507 | 41.45 |
| thủ công | SIFT-BoVW | XGBoost | lr 0.1, depth 6, 300 cây | 44.9 | 44.7 | 47.7 | 44.8 | 45.2 | 47.2 | 58.5 | 4,746 | 5.50 |
| thủ công | SIFT-BoVW | LogReg | C = 0.01 | 43.0 | 43.4 | 45.7 | 43.2 | 45.1 | 45.5 | 0.8 | 4,525 | 0.04 |
| thủ công | SIFT-BoVW | RF | 300 cây, depth None, leaf 3 | 41.5 | 43.2 | 46.8 | 45.5 | 43.7 | 45.6 | 32.6 | 4,617 | 114.64 |

**Zero-shot: macro-F1 trên val theo ngưỡng T** (`data/results/r04/grid_clip_clip_zs.csv`; xác suất softmax cao nhất < T → `unknown`):

| T | 0 (không gán unknown) | 0.5 | **0.6** | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| Val macro-F1 | 56.2 | 60.3 | **60.5** | 60.3 | 59.9 | 59.5 |

Hai giá trị tham khảo T = 0.7 (60.3) và T = 0.8 (59.9) gần như ngang T = 0.6 (60.5): đường cong rất phẳng từ 0.5 đến 0.9; chỉ việc **có** ngưỡng (so với T = 0) là tăng rõ (+4.3).

Nhận xét:
- **Embedding CLIP + bộ phân loại có giám sát vượt xa đặc trưng thủ công**: CLIP + SVM RBF test 88.9 so với 46.4 của SIFT-BoVW + SVM (gần gấp đôi), và cả 4 mô hình trên CLIP đều ≥ 83.3. **Linear probe** đúng nghĩa (bộ phân loại tuyến tính = CLIP + LogReg) đạt 87.6, đã hơn mọi cấu hình thủ công; SVM RBF, RF, XGBoost là bộ phân loại phi tuyến, không phải linear probe. Đây là mức trần tham khảo, kèm rủi ro dữ liệu ở trên.
- **Zero-shot kém hơn nhiều so với có giám sát** (test 58.1 so với 88.9) nhưng vẫn hơn mọi cấu hình thủ công (46.4) mà **không dùng nhãn nào để học** (chỉ chọn T trên val).
- **Mô hình trên CLIP xếp hạng như trên SIFT-BoVW**: SVM > XGBoost ≈ LogReg > RF; SVM thắng với C = 10 (biên lưới).
- **Val khớp test** ở mọi cấu hình CLIP (chênh ≤ 2.4).

### Đánh giá từng lớp (test, `r04`)

P = precision, R = recall. Đầy đủ 9 cấu hình trong notebook mục 4.

| Lớp (số ảnh test) | CLIP + SVM P / R / F1 | CLIP zero-shot P / R / F1 | SIFT-BoVW + SVM P / R / F1 |
|---|---|---|---|
| hold (401) | 79.8 / 84.8 / 82.2 | 69.0 / **10.0** / 17.4 | 39.0 / 27.4 / 32.2 |
| ride (389) | 98.4 / 97.2 / 97.8 | 51.9 / 90.5 / 66.0 | 60.1 / 67.4 / 63.5 |
| eat/drink (208) | 84.9 / 83.7 / 84.3 | 63.2 / 88.5 / 73.7 | 29.5 / 36.1 / 32.5 |
| talk_on_phone (215) | 89.3 / 81.4 / 85.2 | 86.5 / 65.6 / 74.6 | 40.4 / 39.1 / 39.7 |
| work_on_computer (216) | 93.5 / 94.0 / 93.8 | 93.9 / 86.1 / 89.9 | 42.9 / 59.7 / 49.9 |
| unknown (457) | 90.0 / 90.2 / 90.1 | 25.9 / 28.2 / **27.0** | 64.2 / 56.9 / 60.3 |
| **macro** | **89.3 / 88.5 / 88.9** | 65.1 / 61.5 / 58.1 | 46.0 / 47.8 / 46.4 |
| accuracy | **89.2** | 54.7 | 48.8 |

Ma trận nhầm lẫn (số ảnh; hàng = nhãn thật, cột = dự đoán) của **CLIP zero-shot** (T = 0.6):

| Thật \ Dự đoán | hold | ride | eat/drink | talk_on_phone | work_on_computer | unknown |
|---|---|---|---|---|---|---|
| hold | **40** | 22 | 78 | 18 | 9 | 234 |
| ride | 0 | **352** | 0 | 0 | 0 | 37 |
| eat/drink | 2 | 0 | **184** | 2 | 0 | 20 |
| talk_on_phone | 3 | 4 | 11 | **141** | 1 | 55 |
| work_on_computer | 1 | 1 | 4 | 1 | **186** | 23 |
| unknown | 12 | 299 | 14 | 1 | 2 | **129** |

và của **CLIP + SVM**:

| Thật \ Dự đoán | hold | ride | eat/drink | talk_on_phone | work_on_computer | unknown |
|---|---|---|---|---|---|---|
| hold | **340** | 2 | 17 | 13 | 7 | 22 |
| ride | 5 | **378** | 0 | 0 | 0 | 6 |
| eat/drink | 28 | 0 | **174** | 2 | 3 | 1 |
| talk_on_phone | 17 | 2 | 6 | **175** | 1 | 14 |
| work_on_computer | 6 | 0 | 4 | 0 | **203** | 3 |
| unknown | 30 | 2 | 4 | 6 | 3 | **412** |

Nhận xét theo lớp:
- **Zero-shot mạnh ở lớp có vật / hành động rõ** (`work_on_computer` F1 89.9, `talk_on_phone` 74.6, `eat/drink` 73.7) và **hỏng ở 2 lớp "định nghĩa bằng phủ định"**:
  - `unknown` không có prompt riêng: **299/457 ảnh `unknown` (65%) bị đoán là `ride`**. Prompt "a person riding" trở thành lựa chọn mặc định cho người đứng ngoài trời, cạnh xe; softmax giữa 5 prompt luôn chọn 1 lớp, nên ngưỡng T không lọc được các ảnh này (xác suất `ride` vẫn cao).
  - `hold`: **234/401 ảnh (58%) bị gán `unknown`** và 78 thành `eat/drink`. "holding something" quá mơ hồ nên xác suất bị chia đều, rơi dưới ngưỡng.
- **Có nhãn giám sát thì sửa được 2 lớp đó**: recall `hold` / `unknown` là 80.3 / 86.9 với linear probe (LogReg), 84.8 / 90.2 với SVM RBF. Nhãn giám sát cho mô hình học được ranh giới "cầm vật nhưng không làm 3 hành động kia" mà prompt văn bản không diễn đạt được.
- Lỗi còn lại của CLIP + SVM vẫn tập trung ở nhóm "tay cầm vật" (104/204 lỗi, 51%), chủ yếu `eat/drink` → `hold` (28) và `talk_on_phone` → `hold` (17), nhưng số lỗi tuyệt đối giảm từ 966 (SIFT-BoVW + SVM) xuống 204.

### Thời gian chạy và tài nguyên `r04` (bước 2–4 trên `f03`, 24/10/2026)

Lệnh: `LABELS=... DATA_DIR=... CLIP=1 FEATS="clip sift" MODELS="logreg svm rf xgb clip_zs" bash run_all.sh f03 r04 "..."`. Cả pipeline (bước 1–4) **15 phút 37 s** (23:11:29 → 23:27:06), so với ước tính 18–24 phút.

| Bước | Thời gian thực | RAM đỉnh (MB) | RAM tăng thêm (MB) | CPU (s) | Số nhân dùng trung bình |
|---|---|---|---|---|---|
| 2. Chọn mô hình (9 cấu hình) | **453.3 s (7.6 phút)** | 4,864 | 4,715 | 2,770 | 6.1 |
| 3. Test (9 cấu hình, fit trên train+val) | 62.8 s | 865 | 690 | 272 | 4.3 |
| 4. Notebook | ~20 s (ước tính = tổng − bước 1–3) | – | – | – | – |

- **Bước 2 theo cấu hình**: CLIP + XGBoost 102.1 s, CLIP + RF 99.0 s, CLIP + SVM 46.9 s, CLIP + LogReg 3.5 s, **CLIP zero-shot 0.3 s** (6 giá trị T, không train); SIFT-BoVW: SVM 109.5 s, XGBoost 58.5 s, RF 32.6 s, LogReg 0.8 s (sát `r03`).
- CLIP + SVM nhanh hơn SIFT-BoVW + SVM (46.9 s so với 109.5 s) dù cùng ~500 chiều: dữ liệu CLIP tách lớp dễ hơn nên SVM hội tụ nhanh hơn, ít support vector hơn (mô hình 23.9 MB so với 41.5 MB).
- RAM đỉnh bước 2 thấp hơn `r03` (4.9 GB so với 7.8 GB) vì không có HOG (3,780 chiều).
- Dung lượng `data/results/r04/` 261 MB, gần hết là mô hình (RF SIFT-BoVW 114.6 MB, RF CLIP 69.2 MB).

## Kết quả `r03` (4 đặc trưng cơ bản × 4 mô hình, 24/10/2026)

Chạy trên **12,833 ảnh** (`f02`, `labels_split_unk3000.csv`: train 9,069, val 1,878, test 1,886), 4 đặc trưng cơ bản (HOG, LBP, Color-HSV, SIFT-BoVW) × 4 mô hình (LogReg, SVM, RF, XGBoost), **test cả 16 cấu hình**. Cấu hình được chọn là cấu hình tốt nhất **trên val**: **SIFT-BoVW + SVM**. Số test của các cấu hình khác chỉ để báo cáo. Test có `unknown` đã giảm (23% thay vì 56%), nên precision các lớp hành động cao hơn so với phân bố thật. Biểu đồ đầy đủ trong `code/04_report.ipynb`.

Xếp theo val macro-F1. Accuracy chỉ để tham khảo; macro recall = balanced accuracy. RAM là RAM đỉnh khi grid search ở bước 2 (cận trên, gồm nền worker; xem Thời gian chạy `r03` bên dưới).

| Đặc trưng | Mô hình | Siêu tham số tốt nhất | Val macro-F1 | Test macro-F1 | Test accuracy | Test macro precision | Test macro recall | Test weighted F1 | Bước 2 (s) | RAM bước 2 (MB) | Mô hình (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **SIFT-BoVW** | **SVM** | C = 1, γ = scale | **48.0** | 46.4 | 48.8 | 46.0 | 47.8 | 48.4 | 125.5 | 5,670 | 41.45 |
| Color-HSV | XGBoost | lr 0.1, depth 4, 600 cây | 45.8 | 43.2 | 48.1 | 43.1 | 43.8 | 47.2 | 215.4 | 6,848 | 5.81 |
| SIFT-BoVW | XGBoost | lr 0.1, depth 6, 300 cây | 44.9 | 44.7 | 47.7 | 44.8 | 45.2 | 47.2 | 62.5 | 5,655 | 5.50 |
| HOG | SVM | C = 10, γ = scale | 44.5 | **48.3** | **52.6** | **49.4** | **47.9** | **52.1** | 41.6 | 7,106 | 26.39 |
| SIFT-BoVW | LogReg | C = 0.01 | 43.0 | 43.4 | 45.7 | 43.2 | 45.1 | 45.5 | 0.9 | 5,409 | 0.04 |
| LBP | SVM | C = 1, γ = scale | 42.2 | 40.9 | 44.1 | 41.0 | 42.0 | 43.9 | 49.3 | 5,660 | 19.75 |
| SIFT-BoVW | RF | 300 cây, depth None, leaf 3 | 41.5 | 43.2 | 46.8 | 45.5 | 43.7 | 45.6 | 34.8 | 4,933 | 114.64 |
| LBP | XGBoost | lr 0.1, depth 6, 600 cây | 40.9 | 40.1 | 45.3 | 42.3 | 40.2 | 44.0 | 52.9 | 5,829 | 13.06 |
| LBP | LogReg | C = 0.1 | 40.1 | 36.6 | 39.7 | 36.1 | 38.6 | 38.6 | 1.7 | 4,689 | 0.02 |
| HOG | XGBoost | lr 0.05, depth 4, 600 cây | 39.9 | 44.4 | 47.9 | 44.3 | 45.4 | 47.0 | 60.5 | 6,629 | 9.95 |
| Color-HSV | RF | 300 cây, depth None, leaf 3 | 39.3 | 37.6 | 44.0 | 39.8 | 38.8 | 41.4 | 82.3 | 5,972 | 105.71 |
| Color-HSV | SVM | C = 1, γ = scale | 37.8 | 35.9 | 40.2 | 35.7 | 37.0 | 39.2 | 237.1 | 6,412 | 85.50 |
| LBP | RF | 300 cây, depth None, leaf 3 | 37.2 | 34.7 | 41.3 | 38.8 | 35.6 | 39.1 | 50.0 | 6,120 | 107.09 |
| HOG | LogReg | C = 0.01 | 36.2 | 38.7 | 41.1 | 38.7 | 39.2 | 41.5 | 15.0 | 4,513 | 0.27 |
| HOG | RF | 300 cây, depth None, leaf 3 | 33.9 | 35.3 | 43.1 | 44.5 | 35.9 | 40.3 | 295.7 | 7,837 | 98.42 |
| Color-HSV | LogReg | C = 0.01 | 33.5 | 32.7 | 35.5 | 32.7 | 33.3 | 35.7 | 4.6 | 6,153 | 0.07 |

Nhận xét:
- **SVM RBF giúp rõ với SIFT-BoVW**: SIFT-BoVW + SVM đạt test **46.4**, so với 43.4 của SIFT-BoVW + LogReg (cấu hình tốt nhất của `r02`); nhưng không phải mọi mô hình phi tuyến đều hơn (SIFT-BoVW + RF 43.2). SVM là mô hình tốt nhất trên val cho 3/4 đặc trưng (SIFT-BoVW, HOG, LBP); XGBoost tốt nhất cho Color-HSV.
- **Color-HSV mạnh nhất với XGBoost**: Color-HSV + LogReg 33.5 (kém nhất), Color-HSV + XGBoost 45.8 (thứ 2 trên val); RF 39.3, SVM 37.8. Histogram màu không tách tuyến tính được, nhưng XGBoost tìm được ngưỡng trên từng bin màu (bối cảnh: đường, cỏ, bàn).
- **RF kém hơn SVM và XGBoost ở 3/4 đặc trưng** (trừ Color-HSV: Color-HSV + RF 39.3 > Color-HSV + SVM 37.8; sửa 31/10/2026, trước ghi nhầm "ở cả 4 đặc trưng"), lại nặng nhất (98–115 MB mỗi mô hình) và chậm (HOG + RF 296 s ở bước 2).
- **Val dự đoán test tốt**: tương quan hạng Spearman val–test 0.86 trên 16 cấu hình; chênh lệch test − val trung bình −0.2 (từ −3.5 đến +4.6). Riêng **HOG + SVM** đứng thứ 4 trên val (44.5) nhưng cao nhất trên test (48.3). Không chọn lại theo test: chênh 3.5 trên val giữa SIFT-BoVW + SVM và HOG + SVM vẫn là căn cứ duy nhất được phép dùng.
- **Giá trị tốt nhất nằm ở biên lưới**: LogReg C = 0.01 (nhỏ nhất) ở HOG, Color-HSV, SIFT-BoVW; SVM C = 10 (lớn nhất) ở HOG; XGBoost 600 cây (lớn nhất) ở 3/4 đặc trưng. Lưới chưa đủ rộng (xem Cần làm trong README).

### Đánh giá từng lớp (test, `r03`)

Cấu hình chọn theo val (SIFT-BoVW + SVM), so với HOG + SVM (cao nhất trên test) và SIFT-BoVW + LogReg (cấu hình chọn của `r02`). Đầy đủ 16 cấu hình trong notebook mục 4. P = precision, R = recall.

| Lớp (số ảnh test) | SIFT-BoVW + SVM P / R / F1 | HOG + SVM P / R / F1 | SIFT-BoVW + LogReg P / R / F1 |
|---|---|---|---|
| hold (401) | 39.0 / 27.4 / 32.2 | **41.6 / 46.6 / 44.0** | 37.8 / 23.7 / 29.1 |
| ride (389) | 60.1 / 67.4 / 63.5 | **71.8 / 74.6 / 73.1** | 60.6 / 64.5 / 62.5 |
| eat/drink (208) | 29.5 / 36.1 / **32.5** | **30.0** / 30.8 / 30.4 | 28.4 / **37.0** / 32.2 |
| talk_on_phone (215) | 40.4 / **39.1** / **39.7** | **42.8** / 27.4 / 33.4 | 30.4 / 36.7 / 33.3 |
| work_on_computer (216) | 42.9 / **59.7** / **49.9** | **49.5** / 42.1 / 45.5 | 39.5 / 56.9 / 46.7 |
| unknown (457) | **64.2** / 56.9 / 60.3 | 60.4 / **65.9** / **63.0** | 62.5 / 51.9 / 56.7 |
| **macro** | 46.0 / 47.8 / 46.4 | **49.4 / 47.9 / 48.3** | 43.2 / 45.1 / 43.4 |
| accuracy | 48.8 | **52.6** | 45.7 |

Ma trận nhầm lẫn (số ảnh) của **SIFT-BoVW + SVM** (hàng = nhãn thật, cột = dự đoán; `data/results/r03/confusion_sift_svm.csv`):

| Thật \ Dự đoán | hold | ride | eat/drink | talk_on_phone | work_on_computer | unknown |
|---|---|---|---|---|---|---|
| hold | **110** | 54 | 85 | 48 | 55 | 49 |
| ride | 23 | **262** | 14 | 19 | 17 | 54 |
| eat/drink | 36 | 20 | **75** | 27 | 43 | 7 |
| talk_on_phone | 41 | 19 | 29 | **84** | 20 | 22 |
| work_on_computer | 28 | 9 | 27 | 10 | **129** | 13 |
| unknown | 44 | 72 | 24 | 20 | 37 | **260** |

Nhận xét theo lớp:
- **Nhầm giữa 4 lớp "tay cầm vật" vẫn là lỗi chính**: 449/966 lỗi (46%) của SIFT-BoVW + SVM, gần bằng `r02` (48%). Mô hình mạnh hơn không giải quyết được việc đặc trưng toàn ảnh không biết vật gì, ở đâu.
- **HOG + SVM mạnh ở lớp dáng người**: `hold` recall 46.6 (SIFT-BoVW + SVM 27.4), `ride` F1 73.1. HOG mã hoá dáng người (ngồi trên xe/ngựa, tay đưa ra cầm vật); SIFT-BoVW mạnh ở lớp có vật đặc trưng (`work_on_computer`, `talk_on_phone`). Hai đặc trưng bổ sung nhau, nên ghép HOG + SIFT-BoVW (hoặc late fusion 2 SVM) đáng thử (xem Cần làm trong README).
- **`ride` ↔ `unknown`** là lỗi lớn thứ 2 sau nhóm "tay cầm vật", và tính theo cặp hai lớp thì là cặp nhầm qua lại nhiều nhất (72 `unknown` → `ride` + 54 `ride` → `unknown` = 126, sát `hold` ↔ `eat/drink` 121), như `r02`.

### Thời gian chạy `f02` (bước 1, 12,833 ảnh, 24/10/2026)

**Bước 1** (`f02`, 10 khối, `data/features/f02/timing_extract.csv`): tổng thời gian thực **247.2 s**.

| Công đoạn | Thời gian thực (s) |
|---|---|
| Lấy mẫu descriptor + MiniBatchKMeans K=500 (chỉ train) | 83.6 |
| Dense SIFT + mã hoá BoVW (cả 3 split) | 130.5 |
| Đặc trưng ảnh hog/lbp/color/hog_upper/gist/skin (cả 3 split) | 32.2 |
| geom | 0.2 |

- CPU theo khối (cả 12,833 ảnh): sift_descriptor 842 s (65.6 ms/ảnh), bovw_encode 392 s, gist 136 s, LBP 82 s, HOG 28 s (2.2 ms/ảnh).
- So với ước tính trước khi chạy (2–4 phút): thực tế 4 phút 7 s, vượt mức trên. Dense SIFT tốn ~66 ms/ảnh và bovw_encode ~31 ms/ảnh, gấp ~3.5 và ~11 lần `f01` (18.3 và 2.8 ms). **Nguyên nhân chưa xác định.** Không phải do kích thước crop: `sift_desc_pos` luôn resize về cạnh dài 256 px nên số điểm SIFT gần như không đổi. Giả thuyết (chưa kiểm chứng): `cpu_ms_per_img` đo bằng `perf_counter` trong từng tiến trình con, nên khi 10 tiến trình chạy lâu trên 4 nhân nhanh + 6 nhân chậm, cộng với `kmeans.predict` có thể tự mở thêm luồng trong mỗi tiến trình (oversubscription), thời gian mỗi ảnh bị đội lên; `f01` chỉ có 519 ảnh nên ít tranh chấp hơn.
- Các khối ảnh khác (lbp, color, gist, skin, hog_upper, geom) chỉ tốn ~25–30 s thời gian thực; MiniBatchKMeans và dense SIFT chiếm ~87% bước 1.

**Dung lượng `f02`** (đọc từ file; RAM bước 1 chưa đo vì `f02` trích trước khi có `resources.py`):
- Đặc trưng `f02` (cả 3 split, float32): tổng **534 MB**. Theo khối: hog 194.0, sift_spm 128.3, hog_upper 90.6, color 52.6, gist 26.3, sift 25.7, sift_upper 25.7, lbp 12.0, skin 1.1, geom 0.4 MB; codebook MiniBatchKMeans 1.1 MB. Dung lượng tỉ lệ với số chiều: hog (3,780-d) gấp 7.5 lần sift (500-d).

### Thời gian chạy và tài nguyên `r03` (bước 2–4 trên `f02`, 24/10/2026)

Lệnh: `FEATS="hog lbp color sift" MODELS="logreg svm rf xgb" bash run_all.sh f02 r03 "..."` (dùng lại `f02`). Tổng **26 phút 56 s** (22:06:16 → 22:33:12, đo bằng giờ bắt đầu/kết thúc trong log).

| Bước | Thời gian thực | RAM đỉnh (MB) | RAM tăng thêm (MB) | CPU (s) | Số nhân dùng trung bình |
|---|---|---|---|---|---|
| 2. Chọn mô hình (16 cấu hình) | **1,330 s (22.2 phút)** | 7,837 | 7,688 | 8,072 | 6.1 |
| 3. Test (16 cấu hình, fit trên train+val) | 218 s (3.6 phút) | 2,378 | 2,203 | 910 | 4.2 |
| 4. Notebook | ~68 s (ước tính = tổng − bước 2 − bước 3) | – | – | – | – |

- **Bước 2 theo mô hình** (tổng 4 đặc trưng): RF 463 s, SVM 454 s, XGBoost 391 s, LogReg 22 s. Theo đặc trưng: Color-HSV 539 s, HOG 413 s, SIFT-BoVW 224 s, LBP 154 s. Chậm nhất: HOG + RF 296 s, Color-HSV + SVM 237 s, Color-HSV + XGBoost 215 s.
- **Bước 3**: fit lại 16 mô hình mất 202 s (SVM 79 s, XGBoost 68 s, RF 44 s, LogReg 11 s). Dự đoán chậm nhất: Color-HSV + SVM 3.4 ms/ảnh (SVM giữ nhiều support vector); LogReg ≤ 0.02 ms/ảnh.
- **So với ước tính trước khi chạy** (17–25 phút cho cách test cũ, thêm 3–6 phút khi đổi sang test 16 cấu hình): thực tế 27 phút, ở mức trên.
  - Bước 2: 22.2 phút so với ước tính 15–21 phút. SVM bị ước tính thấp nhất: 7.6 phút so với 3.5–5.5 phút (Color-HSV + SVM 237 s, SIFT-BoVW + SVM 126 s; ước tính chỉ dựa trên 1 lần fit C = 1, trong khi C = 10 và 9 cấu hình chạy cùng lúc chậm hơn nhiều). HOG + RF 296 s so với 237.5 s ở `r02` (cùng cấu hình).
  - XGBoost (6.5 phút) và LogReg khớp ước tính.
- **RAM**: đỉnh **7.8 GB** ở HOG + RF, cao hơn nhiều so với ước tính "dưới 3 GB". Đây là cận trên (cộng RSS mọi worker, gồm nền ~10 worker loky và phần mảng memmap dùng chung bị đếm trùng); nguyên nhân chi tiết chưa tách. LogReg cũng có RAM đỉnh 4.5–6.2 GB vì phần lớn là nền của worker còn sống từ cấu hình trước. Máy có 25.8 GB nên chưa có vấn đề, nhưng bộ phối lớn (`fused_all` ~9,300 chiều) cần chú ý.
- **Dung lượng**: `data/results/r03/` 635 MB, trong đó mô hình 634 MB; 4 mô hình RF chiếm 426 MB, 4 SVM 173 MB, 4 XGBoost 34 MB, 4 LogReg 0.4 MB.

## Kết quả `r02` (bộ gộp 4 nguồn, unknown 3000, 24/10/2026)

Chạy trên **12,833 ảnh** (`data/train_test_split/labels_split_unk3000.csv`: train 9,069, val 1,878, test 1,886), 2 đặc trưng (SIFT-BoVW, HOG) × 2 mô hình (LogReg, RF). Biểu đồ đầy đủ trong `code/04_report.ipynb`.
Test có `unknown` đã giảm (23% thay vì 56%), nên precision các lớp hành động cao hơn so với phân bố thật; macro-F1 vẫn so được giữa các cấu hình trong `r02`, nhưng **không so trực tiếp được với `r01`** (khác dữ liệu, khác test).

| Đặc trưng | Mô hình | Siêu tham số tốt nhất | Val macro-F1 | Test macro-F1 | Test balanced acc (= macro recall) | Test macro precision | Test accuracy (tham khảo) |
|---|---|---|---|---|---|---|---|
| SIFT-BoVW (500-d) | LogReg | C = 0.01 | **43.0** | **43.4** | **45.1** | 43.2 | 45.7 |
| SIFT-BoVW (500-d) | RF | 300 cây, depth None, leaf 3 | 41.5 | 43.2 | 43.7 | **45.5** | **46.8** |
| HOG (3,780-d) | LogReg | C = 0.01 | 36.2 | 38.7 | 39.2 | 38.6 | 41.1 |
| HOG (3,780-d) | RF | 300 cây, depth None, leaf 3 | 33.9 | – | – | – | – |

Test macro precision và accuracy tính từ `test_pred_*.npy` (cùng số với `test_report.txt`). Accuracy của SIFT-BoVW + RF cao hơn LogReg dù macro-F1 thấp hơn, vì RF đoán đúng nhiều hơn ở lớp lớn (`unknown`, `ride`): đúng lý do không dùng accuracy để chọn mô hình.

HOG + RF không chạy test vì không phải cấu hình tốt nhất của đặc trưng HOG hay của mô hình RF (`03_final_test.py` chỉ test các cấu hình đó).

Recall từng lớp trên test:

| Cấu hình | hold | ride | eat/drink | talk_on_phone | work_on_computer | unknown |
|---|---|---|---|---|---|---|
| SIFT-BoVW + LogReg | 23.7 | 64.5 | **37.0** | **36.7** | **56.9** | 51.9 |
| SIFT-BoVW + RF | **32.2** | **66.6** | 19.7 | 29.3 | 55.1 | **59.5** |
| HOG + LogReg | 28.7 | 56.3 | 28.4 | 32.1 | 40.3 | 49.5 |

Nhận xét:
- **Val khớp test**: chênh lệch val–test ≤ 3 ở cả 2 cấu hình (với `r01`, val 42 ảnh, có chênh tới 18). Val 1,878 ảnh đủ để chọn mô hình.
- **SIFT-BoVW tốt hơn HOG ở 5/6 lớp**; HOG chỉ hơn ở `hold` (28.7 so với 23.7).
- `hold` là lớp khó nhất (recall 23.7–28.7): "cầm vật" trông giống `unknown` và các lớp có cầm vật khác (`eat/drink`, `talk_on_phone`). `ride` dễ nhất (56–65).
- C tốt nhất là **0.01, giá trị nhỏ nhất của lưới** ở cả 2 đặc trưng: có thể C nhỏ hơn còn tốt hơn (xem Cần làm trong README).
- **RF ngang LogReg về macro-F1 trên SIFT-BoVW** (test 43.2 so với 43.4) nhưng thiên về lớp lớn: recall `hold`, `ride`, `unknown` cao hơn, còn `eat/drink` (19.7) và `talk_on_phone` (29.3) thấp hơn hẳn, dù đã dùng `class_weight="balanced_subsample"`. Trên HOG, RF lệch mạnh hơn nữa (val: recall `talk_on_phone` 7.3, `eat/drink` 11.0).
- RF tốn hơn nhiều: HOG + RF mất 237.5 s ở bước 2, so với 14.6 s của HOG + LogReg.

### Đánh giá từng lớp (test, `r02`)

Tính từ `data/results/r02/test_pred_*.npy` và `data/features/f02/y_test.npy` (cùng số với `test_report.txt` và `confusion_*.png`). P = precision, R = recall.

| Lớp (số ảnh test) | SIFT-BoVW + LogReg P / R / F1 | SIFT-BoVW + RF P / R / F1 | HOG + LogReg P / R / F1 |
|---|---|---|---|
| hold (401) | 37.8 / 23.7 / 29.1 | 36.2 / **32.2** / **34.1** | 34.6 / 28.7 / 31.4 |
| ride (389) | **60.6** / 64.5 / **62.5** | 51.5 / **66.6** / 58.1 | 57.8 / 56.3 / 57.0 |
| eat/drink (208) | 28.4 / **37.0** / **32.2** | 28.1 / 19.7 / 23.2 | 23.3 / 28.4 / 25.6 |
| talk_on_phone (215) | 30.4 / **36.7** / 33.3 | **60.0** / 29.3 / **39.4** | 26.5 / 32.1 / 29.1 |
| work_on_computer (216) | 39.5 / **56.9** / 46.7 | **40.8** / 55.1 / **46.9** | 34.5 / 40.3 / 37.2 |
| unknown (457) | **62.5** / 51.9 / 56.7 | 56.2 / **59.5** / **57.8** | 55.1 / 49.5 / 52.1 |
| **macro** | **43.2 / 45.1 / 43.4** | 45.5 / 43.7 / 43.2 | 38.6 / 39.2 / 38.7 |

Ma trận nhầm lẫn của **SIFT-BoVW + LogReg** (cấu hình tốt nhất trên val; hàng = nhãn thật, cột = dự đoán):

| Thật \ Dự đoán | hold | ride | eat/drink | talk_on_phone | work_on_computer | unknown |
|---|---|---|---|---|---|---|
| hold | **95** | 45 | 93 | 66 | 60 | 42 |
| ride | 23 | **251** | 22 | 27 | 19 | 47 |
| eat/drink | 26 | 14 | **77** | 31 | 49 | 11 |
| talk_on_phone | 40 | 15 | 36 | **79** | 23 | 22 |
| work_on_computer | 18 | 10 | 26 | 19 | **123** | 20 |
| unknown | 49 | 79 | 17 | 38 | 37 | **237** |

Nhận xét theo lớp:
- **Phần lớn lỗi là nhầm giữa 4 lớp "tay cầm vật"** (`hold`, `eat/drink`, `talk_on_phone`, `work_on_computer`): 487/1,024 lỗi (48%) của SIFT-BoVW + LogReg nằm trong nhóm này. Đặc trưng toàn ảnh (SIFT-BoVW là BoVW không giữ vị trí, HOG tính trên cả crop) thấy "có tay, có vật nhỏ" nhưng không phân biệt được vật gì và vật ở đâu so với đầu/tay.
- **`hold` khó nhất** (F1 29–34) và bị nhầm **ra mọi phía**: với SIFT-BoVW + LogReg, 93 ảnh (23%) thành `eat/drink`, 66 (16%) thành `talk_on_phone`, 60 (15%) thành `work_on_computer`. Theo định nghĩa lớp, `hold` là "cầm vật nhưng không làm 3 hành động kia", nên về hình ảnh nó gần như bao trùm 3 lớp đó. Vì `hold` bị dự đoán thành các lớp này nhiều, precision của `eat/drink` (28) và `talk_on_phone` (30) thấp.
- **`eat/drink` hay bị nhầm thành `work_on_computer`** (49/208 = 24%). Giả thuyết (chưa kiểm chứng bằng ảnh): ăn uống thường ở bàn, cùng bối cảnh với ngồi trước máy tính.
- **`ride` ↔ `unknown`** là cặp nhầm lớn thứ 2: 79 ảnh `unknown` (17%) bị đoán là `ride`, 47 ảnh `ride` bị đoán là `unknown`. Giả thuyết (chưa kiểm chứng): người đứng cạnh xe đạp, ngựa, hoặc ở cảnh đường phố có bối cảnh giống `ride`.
- **`work_on_computer` tốt thứ 2** sau `ride` (recall 57): laptop/bàn phím có kết cấu đặc trưng, SIFT-BoVW bắt tốt.
- **RF và LogReg đánh đổi precision/recall khác nhau**: RF có precision `talk_on_phone` cao (60) nhưng recall thấp (29), tức RF chỉ đoán `talk_on_phone` khi rất chắc. Ma trận nhầm lẫn của RF có nhiều ảnh bị đẩy về `ride` và `unknown` hơn (ví dụ `hold` → `ride` 83, `hold` → `unknown` 71), khớp với nhận xét RF thiên về lớp lớn.

### Thời gian chạy và tài nguyên `r02` (bước 2–4 trên `f02`, 24/10/2026)

Thời gian bước 1 (`f02`) xem mục Thời gian chạy của README.

**Bước 2** (`r02`, 2 cấu hình × 4 giá trị C): tổng **16.3 s**. SIFT-BoVW + LogReg: grid 1.5 s, fit lại 0.13 s; HOG + LogReg: grid 11.6 s, fit lại 2.9 s.

**Bước 3** (2 cấu hình): tổng **4.4 s**. Fit trên train+val: SIFT-BoVW 0.15 s, HOG 3.9 s; dự đoán ≤ 0.03 ms/ảnh.

**Bước 2–4 bằng `run_all.sh`** (dùng lại `f02`, đo bằng `time`): **1 phút 14 s**; notebook khoảng 53 s (ước tính = 74 s trừ bước 2–3, gồm cả RF tính importance trên SIFT-BoVW + HOG).

**Bổ sung RF vào `r02`** (`02_train_select.py --update --feats sift hog --models rf`, rồi `03_final_test.py --update` và notebook; đo bằng `time`): **5 phút 24 s**.
- Bước 2: **262.1 s**. SIFT-BoVW + RF: grid 22.8 s + fit lại 1.7 s; HOG + RF: grid **216.5 s** + fit lại 20.9 s.
- Bước 3 (3 cấu hình: SIFT-BoVW + LogReg, HOG + LogReg, SIFT-BoVW + RF): 7.0 s; fit SIFT-BoVW + RF trên train+val 2.3 s.
- Notebook khoảng 55 s (ước tính = 324 s trừ bước 2–3).
- So với ước tính trước khi chạy (5–7 phút; HOG + RF grid 2.5–4.5 phút): khớp. Ước tính dựa trên đo 20 cây trên 1 nhân (SIFT-BoVW ~0.03 s/cây, HOG ~0.28 s/cây).

**Tài nguyên `r02`** (đọc từ file trên đĩa; RAM chưa đo vì `r02` chạy trước khi có `resources.py`):
- Mô hình `r02` (`models/*.joblib`): SIFT-BoVW + LogReg **0.04 MB**, HOG + LogReg 0.27 MB, SIFT-BoVW + RF **114.6 MB**. RF lớn gấp ~3,000 lần LogReg trên cùng đặc trưng (300 cây, `max_depth=None`) mà macro-F1 không hơn; `data/results/r02/` (110 MB) gần như toàn bộ là file RF này.

## Kết quả chạy thử trên mẫu nhỏ (`r01`, thực nghiệm 1–4, 12/10/2026)

Số liệu dưới đây là của phiên bản **`r01`** (`data/results/r01/`; biểu đồ và bảng đầy đủ trong `code/04_report.ipynb`).

Chạy trên **519/15,187 ảnh** (train 261, val 42, test 216), chỉ để kiểm tra pipeline, **không dùng làm kết quả báo cáo**.
Val chỉ có 42 ảnh và không có `ride`, nên việc chọn mô hình còn rất nhiễu.
Mốc so sánh: đoán ngẫu nhiên 6 lớp cho balanced accuracy ≈ 16.7; đoán toàn `unknown` cho macro-F1 ≈ 13.

Val macro-F1 (12 bộ đặc trưng × 4 mô hình):

| Bộ đặc trưng | LogReg | SVM | RF | XGBoost |
|---|---|---|---|---|
| HOG | 21.9 | 22.6 | 15.2 | 15.2 |
| LBP | 14.5 | 22.0 | 15.5 | 20.8 |
| Color-HSV | 21.8 | 32.8 | 15.2 | 14.7 |
| SIFT-BoVW | 33.1 | 34.5 | 26.5 | **35.2** |
| geom | 15.4 | 21.9 | 25.5 | 26.6 |
| sift_spm | 29.6 | 26.5 | 26.5 | 23.4 |
| upper | 23.4 | 31.9 | 15.2 | 15.2 |
| gist | 22.3 | 19.9 | 15.2 | 22.7 |
| skin | 24.2 | 27.2 | 25.0 | 27.8 |
| fused | 31.5 | 15.2 | 15.2 | 15.2 |
| combo_local | 27.5 | 15.2 | 15.2 | 14.9 |
| fused_all | 32.3 | 15.2 | 15.2 | 26.5 |

Test (mô hình tốt nhất trên val của mỗi bộ đặc trưng, train lại trên train+val):

| Bộ đặc trưng | Mô hình | Val macro-F1 | Test macro-F1 | Test balanced acc |
|---|---|---|---|---|
| SIFT-BoVW | XGBoost (tốt nhất trên val) | **35.2** | 17.3 | 17.8 |
| Color-HSV | SVM | 32.8 | 21.6 | 22.2 |
| fused_all | LogReg | 32.3 | 24.8 | **30.6** |
| upper | SVM | 31.9 | 24.4 | 24.4 |
| fused | LogReg | 31.5 | 25.4 | 26.4 |
| sift_spm | LogReg | 29.6 | 16.3 | 16.6 |
| skin | XGBoost | 27.8 | 21.3 | 22.1 |
| combo_local | LogReg | 27.5 | 23.2 | 25.2 |
| geom | XGBoost | 26.6 | **28.3** | 27.9 |
| gist | XGBoost | 22.7 | 19.9 | 19.1 |
| HOG | SVM | 22.6 | 22.6 | 25.3 |
| LBP | SVM | 22.0 | 19.0 | 21.5 |

Ngoài ra bước 3 còn chạy test cho mô hình tốt nhất của từng loại mô hình: SIFT-BoVW + LogReg 26.2, SIFT-BoVW + SVM 21.8, SIFT-BoVW + RF 14.1 (test macro-F1).

Nhận xét (trên mẫu nhỏ):
- **Val không dự đoán được test**: cấu hình tốt nhất trên val (SIFT-BoVW + XGBoost, 35.2) chỉ đạt 17.3 trên test, gần mức đoán ngẫu nhiên. Với 42 ảnh val, chênh lệch 1–2 ảnh đúng/sai đã làm đổi thứ hạng. Cần đủ dữ liệu trước khi kết luận.
- **Phối hợp ổn định hơn đặc trưng đơn lẻ**: `fused`, `fused_all`, `combo_local`, `upper` đều đạt 23–25 trên test; `fused_all` + LogReg có balanced accuracy cao nhất (30.6).
- **`geom` (8 chiều) mạnh bất ngờ**: test macro-F1 cao nhất (28.3), và có importance trung bình mỗi chiều cao nhất trong RF trên `fused_all` (sau đó là `skin`). Tỉ lệ khung và vị trí người là thông tin mà các đặc trưng ảnh (đã resize) bỏ mất.
- **SVM/RF/XGBoost trên bộ phối hợp lớn thường chỉ đoán `unknown`** (15.2) khi ít dữ liệu; LogReg chịu được số chiều cao tốt hơn.
- `ride` đạt recall 0 ở mọi cấu hình (train 8 ảnh, val 0, test 3).

### Thời gian chạy `f01` / `r01` (mẫu 519 ảnh, 12/10/2026)

Máy: Apple Silicon (arm64), 10 CPU, `n_jobs=-1` (như mục [Thời gian chạy](README.md#thời-gian-chạy) của README).

**Bước 1: trích đặc trưng** (`f01`, đo bằng code, `data/features/f01/timing_extract.csv`). Tổng thời gian thực **16.3 s** (`run_info.json`).

| Khối | CPU (s, cả 519 ảnh) | CPU ms / ảnh |
|---|---|---|
| sift_descriptor (dense SIFT) | 9.49 | 18.3 |
| kmeans K=500 (học codebook, chỉ trên train) | 4.53 | (cố định, chỉ train) |
| gist (32 bộ lọc Gabor) | 2.89 | 5.6 |
| LBP | 1.86 | 3.6 |
| bovw_encode (sift + sift_spm + sift_upper) | 1.45 | 2.8 |
| read_image (đọc ảnh, 2 lượt) | 0.96 | 1.9 |
| HOG | 0.72 | 1.4 |
| hog_upper / skin | 0.35 / 0.34 | 0.7 / 0.7 |
| Color-HSV | 0.11 | 0.2 |
| geom (từ bbox, không đọc ảnh) | 0.01 | 0.02 |

Ước tính cho đủ 15,187 ảnh: khoảng 35 ms CPU / ảnh → ~9 phút CPU, chạy song song khoảng **2–3 phút**. Đây là ước tính, chưa đo.

**Bước 2: chọn mô hình (48 cấu hình)**: `r01`, đo bằng code (`data/results/r01/val_results.csv`, `run_info.json`). Tổng thời gian thực **69.1 s**.

| Mô hình | Số lần fit trong grid | Grid search (s) | Fit lại cấu hình tốt nhất trên train (s) |
|---|---|---|---|
| LogReg | 48 | 2.9 | 0.8 |
| SVM | 108 | 4.3 | 1.0 |
| RF | 96 | 20.5 | 3.0 |
| XGBoost | 96 | 18.9 | 17.1 |

- Theo bộ đặc trưng (tổng grid search 4 mô hình): HOG 8.1 s, `fused_all` 6.9 s, `fused` 5.6 s, Color-HSV 5.1 s; rẻ nhất `geom` 0.7 s và `skin` 0.9 s.
- Cấu hình chậm nhất (grid + fit lại): Color-HSV + XGBoost (5.4 s), gist + XGBoost (4.6 s), HOG + RF (4.3 s).
- XGBoost: fit lại 1 lần bằng tất cả CPU (17.1 s cho 12 bộ) tốn ngang cả grid search song song.
- Tổng các cột cộng lại (≈ 68 s) gần bằng thời gian thực; phần còn lại là thời gian nạp dữ liệu và khởi tạo tiến trình.

So với trước khi sửa XGBoost (thực nghiệm 4, đo bằng `time`): toàn bộ bước 2 mất 8 phút 19 s, trong đó XGBoost 461.9 s. Đo riêng 1 lần fit XGBoost trên `fused_all` (~9,300 chiều, 300 cây): `max_bin=256` mất 28.3 s, `max_bin=64` mất 15.0 s.
Với đủ dữ liệu (train lớn gấp ~26 lần), SVM RBF tăng khoảng O(n²), nên bước 2 có thể mất **hàng chục phút đến vài giờ** (ước tính, chưa đo).

**Bước 3: test (15 cấu hình)**: `r01`, tổng thời gian thực **9.0 s** (`run_info.json`).
- Fit trên train+val: tổng 7.4 s; chậm nhất là XGBoost (1.1–1.9 s mỗi cấu hình), LogReg/SVM ≤ 0.21 s.
- Dự đoán: tổng 0.15 s cho 15 × 216 ảnh, ≤ 0.21 ms/ảnh. Thời gian chạy thật vì vậy chủ yếu nằm ở trích đặc trưng (~35 ms CPU/ảnh).

**Bước 4: notebook**: khoảng 16 s (ước tính = tổng `run_all.sh` 110.8 s trừ bước 1–3; gồm cả thời gian khởi động Python).

**Cả pipeline** (`run_all.sh`, đo bằng `time`): **1 phút 51 s**.

## Bài học kinh nghiệm

- **XGBoost chậm theo số cột, không phải theo số ảnh.** Thuật toán `hist` dựng histogram cho từng cột ở mỗi vòng, và multi-class 6 lớp nghĩa là 6 cây mỗi vòng. Vì vậy với bộ 9,300 chiều, 261 ảnh vẫn mất khoảng 28 s/lần fit.
  - Cách xử lý: **giảm chiều bằng PCA (256) trước XGBoost** khi > 1,500 chiều, và giảm `max_bin` xuống 64.
  - Cái giá: cây không còn tách trực tiếp trên từng chiều gốc của SIFT-BoVW/HOG nữa, mà trên thành phần PCA.
  - Áp dụng tương tự cho SVM RBF (cũng PCA 256).
- **Không chạy song song lồng nhau**: `GridSearchCV(n_jobs=-1)` đã song song theo cấu hình, nên mô hình bên trong (RF/XGBoost) để `n_jobs=1`. Chỉ khi fit một mô hình đơn lẻ mới cho dùng hết CPU (`use_all_cores` trong `models.py`).
- **XGBoost không có `class_weight`**: phải truyền `sample_weight=compute_sample_weight("balanced", y)` vào `fit` (qua `clf__sample_weight` trong Pipeline). Train phải có đủ cả 6 lớp, nếu không XGBoost báo lỗi nhãn.
- **Val nhỏ thì không chọn mô hình được**: với 42 ảnh, cấu hình tốt nhất trên val (35.2) chỉ đạt 17.3 trên test. Cần đủ dữ liệu, hoặc chuyển sang cross-validation theo ảnh.
- **Mô hình cây / SVM trên bộ phối hợp lớn dễ sụp về đoán toàn `unknown`** (macro-F1 15.2) khi mỗi lớp chỉ có vài chục ảnh; LogReg ổn định hơn.
- **`class_weight="balanced_subsample"` chưa đủ để RF quan tâm lớp nhỏ**: ở `r02`, RF vẫn thiên về `hold`/`ride`/`unknown` (lớp nhiều ảnh) và bỏ rơi `eat/drink`, `talk_on_phone`. Với đặc trưng nhiều chiều (HOG 3,780-d), lệch còn mạnh hơn. LogReg cân bằng giữa các lớp tốt hơn.
- **Giá trị tốt nhất nằm ở biên lưới thì lưới chưa đủ rộng**: ở `r02`, C = 0.01 (nhỏ nhất) thắng ở cả SIFT-BoVW và hog. Cần mở rộng lưới về phía đó trước khi kết luận.
- **Ước tính thời gian từ mẫu nhỏ có thể thấp hơn thực tế**: ước tính bước 1 cho 12,833 ảnh từ số đo 519 ảnh là 2–4 phút, thực tế 4 phút 7 s; ms/ảnh của SIFT và BoVW tăng 3.5–11 lần khi chạy song song lâu (nguyên nhân chưa xác định, xem [Thời gian chạy](README.md#thời-gian-chạy)). Nên chạy thử trên vài nghìn ảnh trước khi ước tính cho lần chạy lớn.
- **Chọn mô hình phải có luật phá hoà**: với val nhỏ, nhiều cấu hình có cùng macro-F1 (ví dụ SIFT-BoVW + RF và sift_spm + RF cùng 26.462). Sắp xếp không ổn định thì mỗi lần chạy có thể chọn cấu hình khác; `rank_results` trong `models.py` xét macro-F1 → balanced acc → tên.
- **Ước tính thời gian SVM từ 1 lần fit thường thấp hơn thực tế**: ở `r03`, ước tính 3.5–5.5 phút từ 1 lần fit với C = 1, thực tế 7.6 phút. C lớn (10) hội tụ chậm hơn nhiều, và 9 cấu hình chạy cùng lúc trên 4 nhân nhanh + 6 nhân chậm làm mỗi fit chậm thêm. Nên đo cả cấu hình chậm nhất của lưới (C lớn nhất) trước khi ước tính.
- **RAM đỉnh khi grid search song song gấp nhiều lần kích thước dữ liệu**: `r03` đạt 7.8 GB (HOG + RF) trong khi ma trận HOG train+val chỉ 164 MB. Khi ước tính RAM phải tính nền của ~10 worker (mỗi worker đã nạp numpy/sklearn) và số mô hình fit đồng thời (8 rừng 300 cây không giới hạn độ sâu), không chỉ kích thước X.
- **Chênh lệch nhỏ trên val không đáng tin để xếp hạng**: ở `r03`, SIFT-BoVW + SVM hơn HOG + SVM 3.5 trên val nhưng thua 1.9 trên test. Dù val tương quan tốt với test (Spearman 0.86), các cấu hình cách nhau ~3 nên coi là ngang nhau; muốn phân định thì cần cross-validation hoặc nhiều seed.
- **Mô hình tuyến tính đánh giá thấp đặc trưng histogram**: Color-HSV với LogReg kém nhất (33.5) nhưng với XGBoost đứng thứ 2 (45.8). Không nên loại một đặc trưng chỉ vì kết quả với LogReg.
- **Zero-shot cần prompt cho mọi lớp, kể cả lớp "còn lại"**: ở `r04`, `unknown` không có prompt nên softmax giữa 5 prompt hành động luôn chọn 1 lớp; 65% ảnh `unknown` bị đoán `ride` với xác suất cao, ngưỡng T không lọc được. Ngưỡng trên xác suất chỉ bắt được ảnh "mơ hồ", không bắt được ảnh "tự tin nhưng sai lớp".
- **Lớp định nghĩa bằng phủ định khó diễn đạt bằng văn bản**: `hold` ("cầm vật nhưng không ăn/uống, gọi điện, dùng máy tính") có recall zero-shot 10.0 nhưng linear probe (LogReg) 80.3, SVM RBF 84.8. Với lớp kiểu này, cần nhãn giám sát.
- **Ngưỡng T của softmax CLIP phải đặt trên xác suất, không trên cosine**: cosine ảnh–văn bản của CLIP chỉ khoảng 0.2–0.3, nên ngưỡng 0.7/0.8 chỉ có nghĩa với xác suất sau softmax (logit = 100 × cosine).
- **Ước tính tốc độ từ lần chạy thử rất nhỏ bị lệch do chi phí khởi động**: chạy thử CLIP 60 ảnh cho ~60 ảnh/giây (gồm nạp mô hình), thực tế 85 ảnh/giây; ước tính CLIP 3.5–5 phút, thực tế 2.5 phút.

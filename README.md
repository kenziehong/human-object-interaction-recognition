# Human-Object Interaction Recognition (V-COCO, 6 lớp)

Đồ án môn Thị giác máy tính: nhận diện hành động người–vật. Bài toán tham khảo từ DP-HOI ([Li et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Disentangled_Pre-training_for_Human-Object_Interaction_Detection_CVPR_2024_paper.pdf)). Bài báo làm **phát hiện** HOI (định vị người, vật và hành động); đồ án này chỉ làm nhánh **nhận diện hành động** trên ảnh crop người, và dùng phương pháp khác với bài báo. Tóm tắt input, output, metric của bài báo và so sánh với đồ án: [`doc/README_paper.md`](doc/README_paper.md).

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
- Đánh giá: **macro-F1** (chỉ số chính, dùng để chọn mô hình) và **recall từng lớp**; ngoài ra ghi accuracy, precision, F1 (macro, weighted, từng lớp) và confusion matrix. Accuracy chỉ để tham khảo vì `unknown` chiếm đa số. Chọn siêu tham số trên **val**, **test** chỉ dùng để báo cáo.

Mô tả dữ liệu: [`data/vcoco_6class_crops/README.md`](data/vcoco_6class_crops/README.md).

> **Repo không chứa ảnh, đặc trưng hay kết quả** (`data/`, gồm cả `data/features/` và `data/results/`, nằm trong `.gitignore`; trong `data/` chỉ đưa lên file mô tả dữ liệu và các file split trong `data/train_test_split/`). Dữ liệu là ảnh crop người cắt từ [V-COCO](https://github.com/s-gupta/v-coco) (ảnh COCO 2014), cách tạo ghi trong README dữ liệu. Đặt dữ liệu vào `data/vcoco_6class_crops/` theo đúng cấu trúc trong đó (`train/val/test/<lớp>/*.jpg` + `labels.csv`). Các số liệu và phiên bản `fXX` / `rXX` nhắc tới bên dưới là kết quả chạy trên máy của người làm đồ án; biểu đồ và bảng vẫn xem được trong output của `code/04_report.ipynb`.

## Cấu trúc

```
human-object-interaction-recognition/
├── data/vcoco_6class_crops/     dữ liệu crop (train/val/test/<lớp>/*.jpg + labels.csv)
├── data/dataset_6class_crops/   bộ gộp 4 nguồn (<lớp>/*.jpg + labels.csv + labels_ignore.csv), không có cột split
├── data/train_test_split/       file nhãn đã chia split của bộ gộp (labels_split.csv, labels_split_unk3000.csv)
├── data/features/<fXX>/         (tự sinh) đặc trưng .npy, nhãn, codebook SIFT, timing_extract.csv, run_info.json
├── data/results/<rXX>/          (tự sinh) kết quả val/test, confusion matrix, mô hình .joblib, run_info.json
├── code/
│   ├── config.py                đường dẫn, danh sách lớp, tham số đặc trưng
│   ├── data_utils.py            đọc labels.csv, đọc ảnh (bỏ qua ảnh không có trên đĩa)
│   ├── features.py              hog, lbp, color, sift-BoVW (+spm, upper), hog_upper, gist, skin, geom
│   ├── models.py                4 mô hình + lưới siêu tham số + FEATURE_SETS (cách phối)
│   ├── versioning.py            quản lý phiên bản, chặn ghi đè, ghi run_info.json
│   ├── resources.py             đo RAM đỉnh, CPU, dung lượng, thông tin máy
│   ├── clip_feat.py             (tuỳ chọn) embedding ảnh/prompt CLIP ViT-B/32: mốc so sánh, không phải đặc trưng thủ công
│   ├── 01_extract_features.py   trích đặc trưng -> data/features/<fXX>/
│   ├── 02_train_select.py       grid search trên val (12 bộ đặc trưng x 4 mô hình) -> data/results/<rXX>/val_results.csv
│   ├── 03_final_test.py         train lại trên train+val, đánh giá test -> data/results/<rXX>/
│   └── 04_report.ipynb          báo cáo chính: hiển thị phiên bản mới nhất (hiện là r04)
├── doc/                         README_paper.md (tóm tắt input, output, metric của DP-HOI); PDF bài báo chỉ để ở máy, không có trên repo
├── run_all.sh                   chạy tuần tự bước 1 → 4
└── requirements.txt
```

`data/features/` và `data/results/` do script sinh ra, không có trên repo.

## Dữ liệu

Trong `data/` hiện có 2 bộ:

- `data/vcoco_6class_crops/`: bộ **pipeline đang dùng** (`DATA_DIR` trong `code/config.py`). Chỉ gồm V-COCO, đã chia train/val/test. Trên đĩa mới có 519/15.187 ảnh.
- `data/dataset_6class_crops/`: bộ **mới, gộp 4 nguồn** (V-COCO, PASCAL VOC 2012 Action, HICO-DET, Stanford40), đã lọc tay. Pipeline **chưa dùng**. Mô tả chi tiết và quy tắc gán nhãn ở `data/dataset_6class_crops/README.md`.

**Số lượng `dataset_6class_crops`** (kiểm tra ngày 24/10/2026): có 22.366 file `.jpg`, khớp 22.366 dòng trong `labels.csv`, không thiếu file, không trùng đường dẫn.

| Lớp | v_coco | pascal_voc | hico_det | stanford40 | Tổng | Tỉ lệ |
|---|---:|---:|---:|---:|---:|---:|
| hold | 1.416 | 0 | 423 | 862 | 2.701 | 12,1% |
| ride | 853 | 951 | 293 | 767 | 2.864 | 12,8% |
| eat/drink | 1.145 | 0 | 86 | 255 | 1.486 | 6,6% |
| talk_on_phone | 615 | 436 | 20 | 256 | 1.327 | 5,9% |
| work_on_computer | 777 | 427 | 27 | 224 | 1.455 | 6,5% |
| unknown | 9.284 | 939 | 571 | 1.739 | 12.533 | 56,0% |
| **Tổng** | 14.090 | 2.753 | 1.420 | 4.103 | **22.366** | |

- **Không có cột `split`**, chỉ có cột `split_goc` (split của nguồn gốc): v_coco gồm train 6.340, val 1.046, test 6.704; pascal_voc 2.753 ảnh trainval; hico_det 1.420 ảnh train; stanford40 4.103 ảnh để trống.
- `unknown` chiếm 56% toàn bộ, riêng v_coco là 66%. pascal_voc không có `hold` và `eat/drink`.
- `labels_ignore.csv` ghi 10.002 người bị loại (không xuất crop): stanford40 5.388, pascal_voc 3.214, hico_det 1.400. Lý do nhiều nhất là "tay cầm vật ngoài danh sách vật cầm tay" (3.058).
- Thư mục lớp là `eat_drink`, nhưng trong `labels.csv` nhãn vẫn ghi `eat/drink`.

**Chia train/val/test** (24/10/2026): `code/train_test_split.py`, seed 2026, tỉ lệ 70/15/15. Kết quả ghi ra `data/train_test_split/labels_split.csv` (= `labels.csv` + cột `split`); `labels.csv` giữ nguyên. `crop_path` vẫn tính từ `data/dataset_6class_crops/`, nên khi chạy pipeline phải chỉ cả thư mục ảnh: `LABELS=data/train_test_split/labels_split_unk3000.csv DATA_DIR=data/dataset_6class_crops bash run_all.sh ...` (hoặc `01_extract_features.py --labels ... --data_dir ...`).
- Chia **theo ảnh gốc** (`dataset` + `source_image`, 17.304 ảnh): mọi crop của cùng một ảnh nằm chung 1 split. Phân tầng theo lớp × nguồn bằng `StratifiedGroupKFold` 20 fold (14 fold train, 3 val, 3 test).
- **Không** giữ split gốc (`split_goc`) của V-COCO: test mới lấy ảnh từ cả 4 nguồn, nên kết quả trên test này không so trực tiếp được với test V-COCO.
- File `labels_split.csv` giữ đủ 12.533 `unknown`.
- **Bản giảm `unknown`**: `train_test_split.py --max_unknown 3000` → `data/train_test_split/labels_split_unk3000.csv` (12.833 crop). Sau khi chia, các dòng `unknown` được xáo theo seed 2026 và chỉ giữ 3.000 dòng đầu, tính trên cả 3 split. Split của mỗi crop giữ nguyên như trong `labels_split.csv`. Còn lại: train 9.069 / val 1.878 / test 1.886. `unknown` còn 2.087 / 456 / 457, tức 23% thay vì 56%. Vì giảm cả ở test, precision các lớp hành động trên test sẽ cao hơn so với khi giữ đủ `unknown`; khi báo cáo cần ghi rõ.

| Lớp | train | val | test | Tổng |
|---|---:|---:|---:|---:|
| hold | 1.916 | 384 | 401 | 2.701 |
| ride | 2.053 | 422 | 389 | 2.864 |
| eat/drink | 1.078 | 200 | 208 | 1.486 |
| talk_on_phone | 920 | 192 | 215 | 1.327 |
| work_on_computer | 1.015 | 224 | 216 | 1.455 |
| unknown | 8.686 | 1.944 | 1.903 | 12.533 |
| **Tổng** | **15.668 (70,1%)** | **3.366 (15,0%)** | **3.332 (14,9%)** | **22.366** |

Theo nguồn (train / val / test): v_coco 9.866 / 2.124 / 2.100; stanford40 2.869 / 619 / 615; pascal_voc 1.938 / 412 / 403; hico_det 995 / 211 / 214.

## Cài đặt (1 lần)

```bash
git clone https://github.com/kenziehong/human-object-interaction-recognition.git
cd human-object-interaction-recognition
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
brew install libomp        # macOS: XGBoost cần OpenMP
```

`requirements.txt` có `open_clip_torch` (kéo theo `torch`, `torchvision`, vài trăm MB): **chỉ cần cho khối CLIP**. Ai chỉ chạy đặc trưng thủ công có thể bỏ dòng đó; `torch` chỉ được import khi trích với `--clip`. Lần đầu trích CLIP sẽ tải trọng số ViT-B/32 `openai` (~350 MB) vào cache của Hugging Face.

- Thiếu `libomp` thì XGBoost báo `Library not loaded: libomp.dylib`.
- Cảnh báo `RuntimeWarning: ... encountered in matmul` là báo nhầm của numpy 2.0 + Accelerate trên macOS (kết quả vẫn đúng); đã tắt trong `config.py`.

## Cách chạy

Mọi kết quả được lưu theo **phiên bản**, không bao giờ ghi đè hay xoá kết quả cũ:
- `data/features/<fXX>/`: một bộ đặc trưng (đổi tham số đặc trưng hoặc thêm ảnh → phiên bản `f` mới).
- `data/results/<rXX>/`: một lần chọn mô hình + test, dùng một phiên bản đặc trưng (ghi trong `run_info.json`).
- Script **dừng lại** nếu phiên bản đã tồn tại. Muốn bổ sung vào phiên bản cũ phải truyền `--update` một cách chủ động.
- Mỗi phiên bản có `run_info.json`: thời gian, lệnh đã chạy, tham số, ghi chú (`--note`), số ảnh, cấu hình đặc trưng, lưới siêu tham số.
- **Mỗi phiên bản mới phải được thêm vào bảng [Phiên bản kết quả](#phiên-bản-kết-quả).**

**Chạy toàn bộ bằng 1 lệnh** (ở thư mục gốc repo; thời gian xem mục [Thời gian chạy](#thời-gian-chạy)):

```bash
bash run_all.sh                          # tạo data/features/f01 + data/results/r01, rồi chạy notebook
bash run_all.sh f02 r02 "ghi chú"        # phiên bản khác; nếu data/features/f02 đã có thì dùng lại

# Bộ gộp 4 nguồn (file split ở data/train_test_split/, ảnh ở data/dataset_6class_crops/), chỉ chạy một phần:
LABELS=data/train_test_split/labels_split_unk3000.csv DATA_DIR=data/dataset_6class_crops \
  FEATS="sift hog" MODELS="logreg" bash run_all.sh f02 r02 "ghi chú"
```

Biến môi trường tuỳ chọn của `run_all.sh`: `LABELS` (file nhãn có cột `split`), `DATA_DIR` (thư mục ảnh mà `crop_path` tính từ đó; mặc định cạnh file nhãn), `FEATS`, `MODELS` (mặc định: mọi bộ đặc trưng có trong phiên bản đặc trưng × mọi mô hình), `CLIP=1` (bước 1 trích thêm khối `clip`). `LABELS`/`DATA_DIR` chỉ dùng ở bước 1 và được ghi vào `data/features/<fXX>/run_info.json`, nên notebook tự đọc đúng dữ liệu của từng phiên bản.

Chỉ chạy lại notebook `04_report.ipynb` thì **không** tạo ra kết quả: notebook chỉ đọc `data/features/` và `data/results/` do bước 1–3 sinh ra. Repo không chứa hai thư mục này, nên lần đầu phải chạy `run_all.sh` (hoặc bước 1 → 3 bên dưới) trước. Với cùng dữ liệu và thư viện trong `requirements.txt`, kết quả tái lập chính xác (đã kiểm tra: đặc trưng giống từng bit, val/test lệch 0).

Hoặc chạy từng bước trong thư mục `code/` (ví dụ tạo `f02` và `r02`):

```bash
cd code
../.venv/bin/python 01_extract_features.py --version f02 --note "mô tả"                 # bỏ qua nếu dùng lại đặc trưng cũ
../.venv/bin/python 02_train_select.py --feat_version f02 --version r02 --note "mô tả"  # chọn mô hình trên val
../.venv/bin/python 03_final_test.py --version r02                                      # đánh giá test 1 lần
RUN_VERSION=r02 ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 04_report.ipynb  # báo cáo hiển thị r02
```

**Báo cáo kết quả xem ở `code/04_report.ipynb`**: notebook này luôn hiển thị phiên bản mới nhất (hiện là `r04`). Khi có phiên bản mới, sửa giá trị mặc định `VERSION` ở ô đầu rồi Run All (kernel `.venv`). Muốn xem lại phiên bản cũ thì đặt `VERSION` cũ và chạy lại; `data/results/` chỉ chứa kết quả do script sinh ra, không chứa bản sao notebook hay HTML.

Tùy chọn:
- Thử nhanh: `01_extract_features.py --version f_test --limit 60` (lấy khoảng 10 ảnh mỗi lớp mỗi split).
- Chạy thêm một phần vào phiên bản có sẵn: `02_train_select.py --feat_version f02 --version r02 --update --feats sift color --models svm` (chỉ thay các dòng `--feats × --models`, giữ nguyên các dòng khác). Sau đó chạy `03_final_test.py --version r02 --update`.
- Đổi tham số đặc trưng trong `config.py`, sau đó chạy bước 1 với **phiên bản mới**.
- **CLIP (mốc so sánh)**: `01_extract_features.py --clip` trích thêm khối `clip` (embedding ảnh 512 chiều, ảnh crop được đệm thành hình vuông) và `clip_text.npy` (embedding 5 prompt trong `config.CLIP_PROMPTS`). Bộ đặc trưng `clip` chạy được với mọi mô hình (linear probe) và với **`clip_zs`** (zero-shot): softmax(100 × cosine ảnh–prompt) trên 5 lớp hành động, lấy lớp cao nhất; nếu xác suất cao nhất < T thì gán `unknown`. T ∈ {0, 0.5, 0.6, 0.7, 0.8, 0.9} chọn trên val. Ngưỡng này mô phỏng lỏng ngưỡng T chọn Reliable Person Query của DP-HOI (ở bài báo là ngưỡng điểm phát hiện người, ở đây là ngưỡng tin cậy hành động). Kết quả CLIP thuộc nhóm `group = clip`, chọn cấu hình tốt nhất riêng với nhóm đặc trưng thủ công. **Lưu ý dữ liệu**: CLIP train trên ~400 triệu cặp ảnh–văn bản lấy từ web không công bố, không loại trừ được ảnh COCO (có ảnh test V-COCO); chỉ báo cáo như mốc so sánh.

Đầu ra chính (trong `data/results/<rXX>/`, trừ khi ghi khác):
- `val_results.csv`: mọi cấu hình, gồm `val_macro_f1`, `val_balanced_acc` (= macro recall), `val_accuracy`, `val_macro_precision`, `val_weighted_f1`, `precision_/recall_/f1_<lớp>`, siêu tham số tốt nhất. Các cột accuracy / precision / F1 từng lớp có từ `r03` (hàm `eval_metrics` trong `models.py`).
- `test_results.csv`, `test_report.txt`: kết quả test. Từ `r03`, bước 3 **test mọi cấu hình** đã chạy ở bước 2 (`--best_only` để chỉ test cấu hình tốt nhất cho mỗi bộ đặc trưng và mỗi mô hình, như `r01`, `r02`). Cột `group` (`handcrafted` / `clip`) ghi nhóm phương pháp; `is_best_on_val` đánh dấu cấu hình tốt nhất **trên val trong từng nhóm**. Số test của mọi cấu hình chỉ để báo cáo, không dùng để chọn mô hình. Từ `r03`, `test_results.csv` có cùng nhóm cột như val (tiền tố `test_`); phiên bản cũ chỉ có accuracy / precision trong `test_report.txt`, nhưng notebook tự tính lại đủ từ `test_pred_*.npy`.
- `confusion_<feature>_<model>.png` và `.csv`: ma trận nhầm lẫn trên test dạng **số ảnh** (hàng = nhãn thật, cột = dự đoán), có từ `r03`; `r01`/`r02` chỉ có PNG chuẩn hóa theo hàng, nhưng notebook tự vẽ lại dạng số ảnh từ `test_pred_*.npy`.
- `models/*.joblib`, `test_pred_*.npy`, `val_pred_*.npy` (dự đoán trên val của mọi cấu hình, để vẽ confusion matrix trên val; có từ `r03`).
- `grid_<feature>_<model>.csv`: val macro-F1 của **từng** giá trị trong lưới siêu tham số (ví dụ so T = 0.7 và 0.8 của `clip_zs`); có từ `r04`.
- Thời gian chạy (xem mục [Thời gian chạy](#thời-gian-chạy)):
  - `data/features/<fXX>/timing_extract.csv` (bước 1): `wall_s` (thời gian thực của cả công đoạn, chạy song song), `cpu_s` và `cpu_ms_per_img` (tổng thời gian tính của riêng từng khối đặc trưng).
  - `val_results.csv` (bước 2): `n_grid`, `grid_search_s`, `fit_best_s`, `time_s`.
  - `test_results.csv` (bước 3): `dim`, `fit_s` (train trên train+val), `predict_s`, `predict_ms_per_img`.
  - `run_info.json`: `total_wall_s` của mỗi bước.
- Tài nguyên (đo bằng `code/resources.py`, có từ `f03` / `r03`):
  - `run_info.json` mỗi bước: `peak_rss_mb` (RAM đỉnh của cả cây tiến trình, gồm worker joblib), `peak_delta_mb` (phần tăng thêm so với lúc bắt đầu bước), `cpu_s`, `cpu_util` (số nhân dùng trung bình), `machine_info` (RAM máy, số nhân, phiên bản thư viện); bước 1 thêm `disk_mb`, `disk_mb_by_block`; bước 3 thêm `disk_mb`, `models_mb`.
  - `timing_extract.csv` (bước 1): `stage_peak_rss_mb`, `stage_peak_delta_mb`, `stage_cpu_util` (của cả công đoạn, các khối trong cùng công đoạn có cùng giá trị).
  - `val_results.csv` (bước 2): `peak_rss_mb`, `peak_delta_mb`, `cpu_s`, `cpu_util`, `X_mb` (ma trận train+val) cho từng cấu hình.
  - `test_results.csv` (bước 3): `peak_rss_mb`, `peak_delta_mb`, `cpu_util`, `X_mb`, `model_mb` (kích thước `.joblib`).
  - Cách đọc: `peak_rss_mb` cộng RSS mọi tiến trình nên đếm trùng bộ nhớ dùng chung (cận trên), và gồm cả phần nền của các worker loky đang chờ (mỗi worker ~100–200 MB). `peak_delta_mb` của cấu hình đầu tiên còn gồm RAM khởi động worker. Lấy mẫu mỗi 0.2 s, đỉnh ngắn hơn có thể bị bỏ lỡ.

> Nếu trong `data/` thiếu ảnh so với `labels.csv`, pipeline in cảnh báo và chỉ dùng ảnh đang có.

## Phiên bản kết quả

Tra cứu nhanh: mỗi dòng là một thư mục; chi tiết lệnh và tham số nằm trong `run_info.json` của thư mục đó. Từ 24/10/2026, `features/` và `results/` được chuyển vào `data/features/` và `data/results/` (chỉ đổi chỗ, nội dung giữ nguyên). Cột "Thực nghiệm" trỏ tới mục tương ứng trong [`EXPERIMENTS.md`](EXPERIMENTS.md).

**Đặc trưng (`data/features/`)**

| Phiên bản | Ngày | Số ảnh (train/val/test) | Nội dung | Thực nghiệm | Ghi chú |
|---|---|---|---|---|---|
| `f01` | 12/10/2026 | 261 / 42 / 216 | 10 khối: hog, lbp, color, sift, geom, sift_spm, hog_upper, sift_upper, gist, skin | 3 | Tạo bằng `run_all.sh`; có `timing_extract.csv` |
| `f02` | 24/10/2026 | 9.069 / 1.878 / 1.886 | Bộ gộp 4 nguồn, `labels_split_unk3000.csv`; trích đủ 10 khối như `f01` (r02 chỉ dùng `sift`, `hog`) | 5 | 247 s. `LABELS=data/train_test_split/labels_split_unk3000.csv DATA_DIR=data/dataset_6class_crops` |
| `f02_interrupted` | 24/10/2026 | 9.069 / 1.878 / 1.886 | Dở dang: chỉ có meta, geom và các khối ảnh (hog, lbp, color, hog_upper, gist, skin); chưa có SIFT, chưa có `run_info.json` | – | Lần chạy `LABELS=…labels_split_unk3000.csv FEATS="sift hog" MODELS=logreg bash run_all.sh f02 r02` bị dừng ở bước 1 để chuyển file split sang `data/train_test_split/`. Log: `run_log.txt`. **Không dùng** |
| `f03` | 24/10/2026 | 9.069 / 1.878 / 1.886 | Như `f02` + khối `clip` (CLIP ViT-B/32, 512 chiều, ảnh đệm vuông) và `clip_text.npy` (5 prompt); khối thủ công giống từng bit `f02` | 8 | 400.6 s; RAM đỉnh 4.0 GB. `LABELS=data/train_test_split/labels_split_unk3000.csv DATA_DIR=data/dataset_6class_crops CLIP=1` |

**Kết quả (`data/results/`)**

| Phiên bản | Ngày | Đặc trưng | Số ảnh (train/val/test) | Nội dung | Kết quả chính | Thực nghiệm | Ghi chú |
|---|---|---|---|---|---|---|---|
| `r01` | 12/10/2026 | `f01` | 261 / 42 / 216 | 12 bộ đặc trưng × 4 mô hình, test 15 cấu hình, có đo thời gian | val tốt nhất: sift + XGBoost 0.352; test tốt nhất: geom + XGBoost 0.283 | 1–4 | Cấu hình đã gồm mọi thay đổi của thực nghiệm 1–4 |
| `r02_interrupted` | 24/10/2026 | `f02` | 9.069 / 1.878 / 1.886 | Dở dang: bước 2 chỉ xong sift + LogReg (val 0.4298) | – | – | Bị dừng khi đang chạy hog + LogReg (lỗi `TerminatedWorkerError` trong `run_log.txt` là do dừng tay). **Không dùng** |
| `r02` | 24/10/2026 | `f02` | 9.069 / 1.878 / 1.886 | Bộ gộp, unknown 3000; `sift`, `hog` × LogReg, RF (RF bổ sung bằng `--update`) | val tốt nhất: sift + LogReg 0.430; test: sift + LogReg **0.434**, sift + RF 0.432, hog + LogReg 0.387 | 5, 6 | Bước 3 chạy lại sau khi thêm RF; kết quả LogReg không đổi |
| `r03` | 24/10/2026 | `f02` | 9.069 / 1.878 / 1.886 | `hog`, `lbp`, `color`, `sift` × LogReg, SVM, RF, XGB; test cả 16 cấu hình; đủ chỉ số (accuracy, precision, F1, confusion số ảnh); đo RAM | val tốt nhất: sift + SVM 0.480, test **0.464**; test cao nhất: hog + SVM 0.483 (val 0.445) | 7 | 26 phút 56 s; RAM đỉnh 7.8 GB. Bảng đầy đủ 4 đặc trưng thủ công × 4 mô hình |
| **`r04`** | 24/10/2026 | `f03` | 9.069 / 1.878 / 1.886 | `clip` × LogReg, SVM, RF, XGB, zero-shot (`clip_zs`, lưới T); `sift` × 4 mô hình làm mốc; test cả 9 cấu hình | nhóm clip: clip + SVM val 0.878, test **0.889**; zero-shot T = 0.6, test 0.581; nhóm thủ công: sift + SVM test 0.464 | 8 | **Bản mới nhất** (đang hiển thị trong notebook). CLIP là mốc so sánh (rủi ro dữ liệu web). 15 phút 37 s cả pipeline |

Các lần chạy trong thực nghiệm 1–4 không lưu lại file; số liệu của chúng chỉ còn trong [`EXPERIMENTS.md`](EXPERIMENTS.md). `r01` tái lập đúng kết quả cuối của thực nghiệm 4.
Phiên bản tiếp theo nên đặt là `r05` (và `f04` nếu đổi dữ liệu, tham số đặc trưng hoặc thêm khối).

## Kết quả `r04` (CLIP so với sift, 24/10/2026)

README chỉ giữ kết quả mới nhất. Kết quả và thời gian chạy của các phiên bản cũ nằm trong [`EXPERIMENTS.md`](EXPERIMENTS.md): **`r03` là bảng đầy đủ của 4 đặc trưng thủ công × 4 mô hình** (hog + SVM cao nhất trên test 0.483), `r02`, `r01`.

Chạy trên **12.833 ảnh** (`f03` = các khối của `f02` + khối `clip`; khối thủ công của `f03` giống từng bit `f02`), 2 bộ đặc trưng × mô hình, **test cả 9 cấu hình**:
- **Nhóm `clip`** (mốc so sánh, không phải đặc trưng thủ công): embedding ảnh CLIP ViT-B/32 (512 chiều) × LogReg, SVM, RF, XGB (linear probe / học có giám sát) và `clip_zs` (zero-shot).
- **Nhóm `handcrafted`**: `sift` × 4 mô hình, đặt cạnh để so (số liệu trùng `r03`).

> **Lưu ý dữ liệu**: CLIP được train trên ~400 triệu cặp ảnh–văn bản lấy từ web, không công bố. Không loại trừ được việc dữ liệu đó có ảnh COCO, kể cả ảnh test V-COCO (trong test của `r04`, 1.025/1.886 ảnh là từ v_coco, tức ảnh COCO). Vì vậy kết quả CLIP chỉ là **mốc so sánh**, không thay kết quả chính của đặc trưng thủ công.

Xếp theo nhóm, rồi theo val macro-F1. Cấu hình chọn theo val trong từng nhóm: **clip + SVM** và **sift + SVM**. Accuracy chỉ để tham khảo; macro recall = balanced accuracy. RAM là RAM đỉnh khi grid search ở bước 2 (cận trên, gồm nền worker).

| Nhóm | Đặc trưng | Mô hình | Siêu tham số tốt nhất | Val macro-F1 | Test macro-F1 | Test accuracy | Test macro precision | Test macro recall | Test weighted F1 | Bước 2 (s) | RAM bước 2 (MB) | Mô hình (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| clip | **clip** | **SVM** | C = 10, γ = scale | **0.878** | **0.889** | **0.892** | **0.893** | **0.885** | **0.892** | 46.9 | 3.942 | 23.90 |
| clip | clip | XGB | lr 0.1, depth 4, 600 cây | 0.870 | 0.882 | 0.885 | 0.887 | 0.879 | 0.886 | 102.1 | 4.864 | 5.68 |
| clip | clip | LogReg | C = 0.01 | 0.870 | 0.876 | 0.880 | 0.872 | 0.881 | 0.880 | 3.5 | 1.441 | 0.04 |
| clip | clip | RF | 300 cây, depth None, leaf 3 | 0.847 | 0.833 | 0.836 | 0.865 | 0.820 | 0.839 | 99.0 | 4.295 | 69.21 |
| clip | clip | **zero-shot** | T = 0.6 | 0.605 | 0.581 | 0.547 | 0.651 | 0.615 | 0.508 | 0.3 | 4.255 | 0.01 |
| handcrafted | **sift** | **SVM** | C = 1, γ = scale | 0.480 | 0.464 | 0.488 | 0.460 | 0.478 | 0.484 | 109.5 | 4.507 | 41.45 |
| handcrafted | sift | XGB | lr 0.1, depth 6, 300 cây | 0.449 | 0.447 | 0.477 | 0.448 | 0.452 | 0.472 | 58.5 | 4.746 | 5.50 |
| handcrafted | sift | LogReg | C = 0.01 | 0.430 | 0.434 | 0.457 | 0.432 | 0.451 | 0.455 | 0.8 | 4.525 | 0.04 |
| handcrafted | sift | RF | 300 cây, depth None, leaf 3 | 0.415 | 0.432 | 0.468 | 0.455 | 0.437 | 0.456 | 32.6 | 4.617 | 114.64 |

**Zero-shot: macro-F1 trên val theo ngưỡng T** (`data/results/r04/grid_clip_clip_zs.csv`; xác suất softmax cao nhất < T → `unknown`):

| T | 0 (không gán unknown) | 0.5 | **0.6** | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| Val macro-F1 | 0.562 | 0.603 | **0.605** | 0.604 | 0.599 | 0.595 |

Hai giá trị tham khảo T = 0.7 (0.604) và T = 0.8 (0.599) gần như ngang T = 0.6 (0.605): đường cong rất phẳng từ 0.5 đến 0.9; chỉ việc **có** ngưỡng (so với T = 0) là tăng rõ (+0.04).

Nhận xét:
- **Linear probe CLIP vượt xa đặc trưng thủ công**: clip + SVM test 0.889 so với 0.464 của sift + SVM (gần gấp đôi), và cả 4 mô hình trên clip đều ≥ 0.83. Ngay LogReg trên clip (0.876) đã hơn mọi cấu hình thủ công. Đây là mức trần tham khảo, kèm rủi ro dữ liệu ở trên.
- **Zero-shot kém hơn nhiều so với có giám sát** (test 0.581 so với 0.889) nhưng vẫn hơn mọi cấu hình thủ công (0.464) mà **không dùng nhãn nào để học** (chỉ chọn T trên val).
- **Mô hình trên clip xếp hạng như trên sift**: SVM > XGB ≈ LogReg > RF; SVM thắng với C = 10 (biên lưới).
- **Val khớp test** ở mọi cấu hình clip (chênh ≤ 0.024).

### Đánh giá từng lớp (test, `r04`)

P = precision, R = recall. Đầy đủ 9 cấu hình trong notebook mục 4.

| Lớp (số ảnh test) | clip + SVM P / R / F1 | clip zero-shot P / R / F1 | sift + SVM P / R / F1 |
|---|---|---|---|
| hold (401) | 0.798 / 0.848 / 0.822 | 0.690 / **0.100** / 0.174 | 0.390 / 0.274 / 0.322 |
| ride (389) | 0.984 / 0.972 / 0.978 | 0.519 / 0.905 / 0.660 | 0.601 / 0.674 / 0.635 |
| eat/drink (208) | 0.849 / 0.837 / 0.843 | 0.632 / 0.885 / 0.737 | 0.295 / 0.361 / 0.325 |
| talk_on_phone (215) | 0.893 / 0.814 / 0.852 | 0.865 / 0.656 / 0.746 | 0.404 / 0.391 / 0.397 |
| work_on_computer (216) | 0.935 / 0.940 / 0.938 | 0.939 / 0.861 / 0.899 | 0.429 / 0.597 / 0.499 |
| unknown (457) | 0.900 / 0.902 / 0.901 | 0.259 / 0.282 / **0.270** | 0.642 / 0.569 / 0.603 |
| **macro** | **0.893 / 0.885 / 0.889** | 0.651 / 0.615 / 0.581 | 0.460 / 0.478 / 0.464 |
| accuracy | **0.892** | 0.547 | 0.488 |

Ma trận nhầm lẫn (số ảnh; hàng = nhãn thật, cột = dự đoán) của **clip zero-shot** (T = 0.6):

| Thật \ Dự đoán | hold | ride | eat/drink | talk_on_phone | work_on_computer | unknown |
|---|---|---|---|---|---|---|
| hold | **40** | 22 | 78 | 18 | 9 | 234 |
| ride | 0 | **352** | 0 | 0 | 0 | 37 |
| eat/drink | 2 | 0 | **184** | 2 | 0 | 20 |
| talk_on_phone | 3 | 4 | 11 | **141** | 1 | 55 |
| work_on_computer | 1 | 1 | 4 | 1 | **186** | 23 |
| unknown | 12 | 299 | 14 | 1 | 2 | **129** |

và của **clip + SVM**:

| Thật \ Dự đoán | hold | ride | eat/drink | talk_on_phone | work_on_computer | unknown |
|---|---|---|---|---|---|---|
| hold | **340** | 2 | 17 | 13 | 7 | 22 |
| ride | 5 | **378** | 0 | 0 | 0 | 6 |
| eat/drink | 28 | 0 | **174** | 2 | 3 | 1 |
| talk_on_phone | 17 | 2 | 6 | **175** | 1 | 14 |
| work_on_computer | 6 | 0 | 4 | 0 | **203** | 3 |
| unknown | 30 | 2 | 4 | 6 | 3 | **412** |

Nhận xét theo lớp:
- **Zero-shot mạnh ở lớp có vật / hành động rõ** (`work_on_computer` F1 0.899, `talk_on_phone` 0.746, `eat/drink` 0.737) và **hỏng ở 2 lớp "định nghĩa bằng phủ định"**:
  - `unknown` không có prompt riêng: **299/457 ảnh `unknown` (65%) bị đoán là `ride`**. Prompt "a person riding" trở thành lựa chọn mặc định cho người đứng ngoài trời, cạnh xe; softmax giữa 5 prompt luôn chọn 1 lớp, nên ngưỡng T không lọc được các ảnh này (xác suất `ride` vẫn cao).
  - `hold`: **234/401 ảnh (58%) bị gán `unknown`** và 78 thành `eat/drink`. "holding something" quá mơ hồ nên xác suất bị chia đều, rơi dưới ngưỡng.
- **Linear probe sửa đúng 2 lớp đó**: `hold` recall 0.848, `unknown` 0.902. Nhãn giám sát cho mô hình học được ranh giới "cầm vật nhưng không làm 3 hành động kia" mà prompt văn bản không diễn đạt được.
- Lỗi còn lại của clip + SVM vẫn tập trung ở nhóm "tay cầm vật" (104/204 lỗi, 51%), chủ yếu `eat/drink` → `hold` (28) và `talk_on_phone` → `hold` (17), nhưng số lỗi tuyệt đối giảm từ 966 (sift + SVM) xuống 204.

## Thời gian chạy

Máy: Apple Silicon (arm64), 10 CPU (4 nhân hiệu năng cao + 6 nhân tiết kiệm điện), RAM 24 GB, `n_jobs=-1`. Chỉ ghi phiên bản mới nhất; thời gian `f02` / `r03`, `r02` và `f01` / `r01` xem [`EXPERIMENTS.md`](EXPERIMENTS.md).

### `f03` (bước 1, 12.833 ảnh, 24/10/2026)

**Bước 1** (`f03`, 10 khối thủ công + `clip`, `--clip`): tổng thời gian thực **400.6 s (6.7 phút)**; RAM đỉnh **4.008 MB** (tăng thêm 3.833 MB), CPU 2.660 s (6.6 nhân trung bình); đĩa 585.6 MB.

| Công đoạn | Thời gian thực (s) |
|---|---|
| Lấy mẫu descriptor + KMeans K=500 (chỉ train) | 77.1 |
| Dense SIFT + BoVW (cả 3 split) | 137.3 |
| Đặc trưng ảnh hog/lbp/color/hog_upper/gist/skin (cả 3 split) | 29.3 |
| geom | 0.2 |
| **CLIP ViT-B/32** (cả 3 split, CPU, 8 luồng torch) | **151.7** (train 106.9, val 22.0, test 22.8) |

- CLIP: ~85 ảnh/giây (9.069 ảnh train trong 107 s), RAM đỉnh công đoạn 3.9 GB, 2.2 nhân trung bình. Khối `clip` 26.3 MB trên đĩa (512 chiều float32). Lần đầu tải trọng số ~350 MB từ Hugging Face (thời gian tải nằm trong 151.7 s, không tách riêng).
- So với ước tính trước khi chạy (thủ công ~4 phút + CLIP 3.5–5 phút): thực tế 6.7 phút, CLIP nhanh hơn ước tính (2.5 phút). Ước tính CLIP dựa trên lần chạy thử 60 ảnh, gồm cả thời gian nạp mô hình nên tốc độ bị đánh giá thấp.
- Khối thủ công của `f03` **giống từng bit** `f02` (đã so `sift`, `hog` trên test): pipeline tái lập được.

### `r04` (bước 2–4 trên `f03`, 24/10/2026)

Lệnh: `LABELS=... DATA_DIR=... CLIP=1 FEATS="clip sift" MODELS="logreg svm rf xgb clip_zs" bash run_all.sh f03 r04 "..."`. Cả pipeline (bước 1–4) **15 phút 37 s** (23:11:29 → 23:27:06), so với ước tính 18–24 phút.

| Bước | Thời gian thực | RAM đỉnh (MB) | RAM tăng thêm (MB) | CPU (s) | Số nhân dùng trung bình |
|---|---|---|---|---|---|
| 2. Chọn mô hình (9 cấu hình) | **453.3 s (7.6 phút)** | 4.864 | 4.715 | 2.770 | 6.1 |
| 3. Test (9 cấu hình, fit trên train+val) | 62.8 s | 865 | 690 | 272 | 4.3 |
| 4. Notebook | ~20 s (ước tính = tổng − bước 1–3) | – | – | – | – |

- **Bước 2 theo cấu hình**: clip + XGB 102.1 s, clip + RF 99.0 s, clip + SVM 46.9 s, clip + LogReg 3.5 s, **clip zero-shot 0.3 s** (6 giá trị T, không train); sift: SVM 109.5 s, XGB 58.5 s, RF 32.6 s, LogReg 0.8 s (sát `r03`).
- clip + SVM nhanh hơn sift + SVM (46.9 s so với 109.5 s) dù cùng ~500 chiều: dữ liệu CLIP tách lớp dễ hơn nên SVM hội tụ nhanh hơn, ít support vector hơn (mô hình 23.9 MB so với 41.5 MB).
- RAM đỉnh bước 2 thấp hơn `r03` (4.9 GB so với 7.8 GB) vì không có hog (3.780 chiều).
- Dung lượng `data/results/r04/` 261 MB, gần hết là mô hình (RF sift 114.6 MB, RF clip 69.2 MB).

## Thực nghiệm

Chỉ ghi **thực nghiệm mới nhất**. Toàn bộ nhật ký (mục 1–8), kết luận tạm và [bài học kinh nghiệm](EXPERIMENTS.md#bài-học-kinh-nghiệm) nằm trong [`EXPERIMENTS.md`](EXPERIMENTS.md).

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

## Cần làm

**Trạng thái hiện tại (để tiếp tục lần sau):**
- `.venv` đã cài đủ thư viện, `libomp` đã cài.
- `data/features/f02/` có đủ 10 khối đặc trưng cho bộ gộp (12.833 ảnh, `labels_split_unk3000.csv`). **Không cần chạy lại bước 1** khi chỉ đổi mô hình hoặc lưới: chạy `02_train_select.py --feat_version f02 --version r03 ...`.
- `data/features/f03/` = `f02` + khối `clip`. `data/results/r04/` là **kết quả mới nhất** (clip × 5 mô hình + sift × 4); notebook đang hiển thị `r04`. Bảng đầy đủ đặc trưng thủ công là `r03`.
- `torch` + `open_clip_torch` đã cài trong `.venv` (chỉ thêm gói mới, numpy/sklearn/xgboost giữ nguyên phiên bản).
- `data/features/f01/`, `data/results/r01/` là bản cũ trên 519 ảnh V-COCO, giữ để tham khảo.

Việc bắt buộc, làm trước:
- [ ] **Lấy đủ dữ liệu**: hiện chỉ có 519/15.187 crop trong `data/vcoco_6class_crops/`. Kiểm tra bằng `find data/vcoco_6class_crops/train -name '*.jpg' | wc -l`, phải ra 6803.
- [ ] **Quyết định có chuyển sang `data/dataset_6class_crops/` (22.366 ảnh, 4 nguồn) hay không** (xem mục Dữ liệu). Bộ này có số ảnh hành động nhiều hơn hẳn bộ V-COCO, nhưng chưa có split, nên muốn dùng thì phải:
  - [x] tạo cột `split`: đã có `labels_split.csv` (70/15/15 theo ảnh gốc, seed 2026; xem mục Dữ liệu). Cách này không giữ test gốc của v_coco;
  - kiểm tra xem có ảnh nào ở nguồn khác trùng với ảnh test của v_coco không, tránh lộ test. README dữ liệu chỉ ghi đã bỏ các cặp trùng *cùng một người*;
  - [x] trỏ pipeline sang dữ liệu mới và tạo phiên bản mới: đã chạy `f02`/`r02` (thực nghiệm 5).
- [x] Chạy bước 1 → 4 thành phiên bản `f01` / `r01` (`run_all.sh`).
- [x] Thêm phần "Thời gian chạy" vào `04_report.ipynb` (mục 5).
- [ ] Sau khi có đủ dữ liệu: cập nhật các bảng kết quả, bảng thời gian và nhận xét trong README; viết lại phần "Nhận xét gợi ý" (mục 7) trong notebook theo kết quả thật. Notebook đã chạy với `r02`, nhưng các ô chữ (markdown) vẫn là nhận xét của `r01`.

Cải thiện tiếp từ `r03`:
- [x] **Chạy CLIP (`f03`, `r04`)** (xong, thực nghiệm 8): linear probe clip + SVM test 0.889, zero-shot 0.581.
- [ ] **Zero-shot: thêm prompt cho `unknown`** (ví dụ "a photo of a person standing", "a photo of a person walking") và **prompt ensembling** (nhiều câu mỗi lớp, lấy trung bình embedding). Lý do: 65% ảnh `unknown` bị đoán `ride` vì không có prompt `unknown`; ngưỡng T không sửa được. Rẻ: chỉ cần tính lại `clip_text.npy`, không cần trích lại ảnh.
- [ ] **Ghép CLIP với đặc trưng thủ công** (nối `clip + hog`, hoặc late fusion clip + SVM với hog + SVM): đặc trưng thủ công có bổ sung gì cho CLIP không? Đây là câu hỏi duy nhất còn ý nghĩa cho phần thủ công khi đặt cạnh CLIP.
- [ ] **Kiểm tra rủi ro CLIP đã thấy ảnh test**: so kết quả clip + SVM theo nguồn ảnh test (v_coco/COCO so với stanford40, pascal_voc, hico_det). Nếu v_coco cao bất thường so với nguồn khác thì có dấu hiệu lộ dữ liệu.
- [x] **Đo RAM**: bước 2–3 đã đo ở `r03` (đỉnh 7.8 GB ở hog + RF). Còn thiếu RAM bước 1: sẽ có khi trích `f03`.
- [ ] **Tách nguồn RAM của bước 2** (nền worker, bản sao ma trận, cây RF, cache kernel SVM) trước khi chạy bộ phối lớn (`fused_all` ~9.300 chiều): RAM đỉnh `r03` 7.8 GB với hog 3.780 chiều.
- [x] **Chạy `r03` để có đủ chỉ số trong csv** (xong, thực nghiệm 7): code đã ghi accuracy, precision, F1 từng lớp (val và test) và confusion matrix số đếm (`confusion_*.csv`), nhưng chưa chạy. Có thể gộp vào lần chạy `r03` kế tiếp (ví dụ mở rộng lưới C bên dưới). Notebook đã sửa (tính đủ chỉ số và vẽ confusion số đếm cho mọi cấu hình đã test, chạy được cả với `r01`/`r02`) nhưng output đang lưu vẫn là bản cũ; Run All khi có `r03`.
- [ ] **Mở rộng lưới ở biên**: LogReg C về phía nhỏ (0.001, 0.003; C = 0.01 nhỏ nhất thắng ở hog, color, sift), SVM C về phía lớn (30, 100; C = 10 thắng ở hog), XGBoost thêm 1.000 cây (600 thắng ở 3/4 đặc trưng). LogReg rẻ (~20 s); SVM, XGB thêm vài phút mỗi đặc trưng (ước tính).
- [ ] **Ghép hog + sift cho SVM** (nối đặc trưng, PCA 256 vì > 1.500 chiều) hoặc **late fusion hog + SVM và sift + SVM**: ở `r03`, hog + SVM mạnh ở `hold`, `ride` (dáng người), sift + SVM mạnh ở `talk_on_phone`, `work_on_computer` (vật). Chọn trọng số trên val.
- [ ] Cải thiện lớp `hold` (recall test 0.24–0.29 với LogReg ở `r02`; 0.274 với sift + SVM, 0.466 với hog + SVM ở `r03`). Đã xem confusion (mục Đánh giá từng lớp): `hold` bị nhầm đều sang `eat/drink`, `talk_on_phone`, `work_on_computer`, và 48% lỗi nằm trong 4 lớp "tay cầm vật". Hướng thử: đặc trưng giữ vị trí vật so với đầu/tay (SIFT spatial pyramid, `upper` thân trên) thay cho BoVW toàn ảnh. Lý do: lỗi đến từ việc không phân biệt được vật ở đâu, không phải do thiếu dữ liệu.
- [ ] Xem bằng mắt các ảnh `eat/drink` → `work_on_computer` và `unknown` → `ride` để kiểm chứng giả thuyết "do bối cảnh" (bàn ăn/bàn làm việc, người đứng cạnh xe).
- [ ] ~~Late fusion sift + LogReg và sift + RF~~: thay bằng late fusion hog + SVM và sift + SVM ở trên (RF kém hơn SVM ở `r03`).

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

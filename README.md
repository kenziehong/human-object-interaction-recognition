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
- Đánh giá bằng **macro-F1** và **recall từng lớp** (không dùng accuracy). Chọn siêu tham số trên **val**, **test** chỉ dùng để báo cáo.

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
│   ├── 01_extract_features.py   trích đặc trưng -> data/features/<fXX>/
│   ├── 02_train_select.py       grid search trên val (12 bộ đặc trưng x 4 mô hình) -> data/results/<rXX>/val_results.csv
│   ├── 03_final_test.py         train lại trên train+val, đánh giá test -> data/results/<rXX>/
│   └── 04_report.ipynb          báo cáo chính: hiển thị phiên bản mới nhất (hiện là r01)
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

- Thiếu `libomp` thì XGBoost báo `Library not loaded: libomp.dylib`.
- Cảnh báo `RuntimeWarning: ... encountered in matmul` là báo nhầm của numpy 2.0 + Accelerate trên macOS (kết quả vẫn đúng); đã tắt trong `config.py`.

## Cách chạy

Mọi kết quả được lưu theo **phiên bản**, không bao giờ ghi đè hay xoá kết quả cũ:
- `data/features/<fXX>/`: một bộ đặc trưng (đổi tham số đặc trưng hoặc thêm ảnh → phiên bản `f` mới).
- `data/results/<rXX>/`: một lần chọn mô hình + test, dùng một phiên bản đặc trưng (ghi trong `run_info.json`).
- Script **dừng lại** nếu phiên bản đã tồn tại. Muốn bổ sung vào phiên bản cũ phải truyền `--update` một cách chủ động.
- Mỗi phiên bản có `run_info.json`: thời gian, lệnh đã chạy, tham số, ghi chú (`--note`), số ảnh, cấu hình đặc trưng, lưới siêu tham số.
- **Mỗi phiên bản mới phải được thêm vào bảng [Phiên bản kết quả](#phiên-bản-kết-quả).**

**Chạy toàn bộ bằng 1 lệnh** (ở thư mục gốc repo, khoảng 2 phút trên 519 ảnh):

```bash
bash run_all.sh                          # tạo data/features/f01 + data/results/r01, rồi chạy notebook
bash run_all.sh f02 r02 "ghi chú"        # phiên bản khác; nếu data/features/f02 đã có thì dùng lại

# Bộ gộp 4 nguồn (file split ở data/train_test_split/, ảnh ở data/dataset_6class_crops/), chỉ chạy một phần:
LABELS=data/train_test_split/labels_split_unk3000.csv DATA_DIR=data/dataset_6class_crops \
  FEATS="sift hog" MODELS="logreg" bash run_all.sh f02 r02 "ghi chú"
```

Biến môi trường tuỳ chọn của `run_all.sh`: `LABELS` (file nhãn có cột `split`), `DATA_DIR` (thư mục ảnh mà `crop_path` tính từ đó; mặc định cạnh file nhãn), `FEATS`, `MODELS` (mặc định: tất cả). `LABELS`/`DATA_DIR` chỉ dùng ở bước 1 và được ghi vào `data/features/<fXX>/run_info.json`, nên notebook tự đọc đúng dữ liệu của từng phiên bản.

Chỉ chạy lại notebook `04_report.ipynb` thì **không** tạo ra kết quả: notebook chỉ đọc `data/features/` và `data/results/` do bước 1–3 sinh ra. Repo không chứa hai thư mục này, nên lần đầu phải chạy `run_all.sh` (hoặc bước 1 → 3 bên dưới) trước. Với cùng dữ liệu và thư viện trong `requirements.txt`, kết quả tái lập chính xác (đã kiểm tra: đặc trưng giống từng bit, val/test lệch 0).

Hoặc chạy từng bước trong thư mục `code/` (ví dụ tạo `f02` và `r02`):

```bash
cd code
../.venv/bin/python 01_extract_features.py --version f02 --note "mô tả"                 # bỏ qua nếu dùng lại đặc trưng cũ
../.venv/bin/python 02_train_select.py --feat_version f02 --version r02 --note "mô tả"  # chọn mô hình trên val
../.venv/bin/python 03_final_test.py --version r02                                      # đánh giá test 1 lần
RUN_VERSION=r02 ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 04_report.ipynb  # báo cáo hiển thị r02
```

**Báo cáo kết quả xem ở `code/04_report.ipynb`**: notebook này luôn hiển thị phiên bản mới nhất (hiện là `r02`). Khi có phiên bản mới, sửa giá trị mặc định `VERSION` ở ô đầu rồi Run All (kernel `.venv`). Muốn xem lại phiên bản cũ thì đặt `VERSION` cũ và chạy lại; `data/results/` chỉ chứa kết quả do script sinh ra, không chứa bản sao notebook hay HTML.

Tùy chọn:
- Thử nhanh: `01_extract_features.py --version f_test --limit 60` (lấy khoảng 10 ảnh mỗi lớp mỗi split).
- Chạy thêm một phần vào phiên bản có sẵn: `02_train_select.py --feat_version f02 --version r02 --update --feats sift color --models svm` (chỉ thay các dòng `--feats × --models`, giữ nguyên các dòng khác). Sau đó chạy `03_final_test.py --version r02 --update`.
- Đổi tham số đặc trưng trong `config.py`, sau đó chạy bước 1 với **phiên bản mới**.

Đầu ra chính (trong `data/results/<rXX>/`, trừ khi ghi khác):
- `val_results.csv`: macro-F1, balanced accuracy, recall từng lớp, siêu tham số tốt nhất của mọi cấu hình.
- `test_results.csv`, `test_report.txt`: kết quả test của mô hình tốt nhất cho mỗi bộ đặc trưng và cho mỗi mô hình (cột `is_best_on_val` đánh dấu mô hình tốt nhất trên val).
- `confusion_<feature>_<model>.png`, `models/*.joblib`, `test_pred_*.npy`.
- Thời gian chạy (xem mục [Thời gian chạy](#thời-gian-chạy)):
  - `data/features/<fXX>/timing_extract.csv` (bước 1): `wall_s` (thời gian thực của cả công đoạn, chạy song song), `cpu_s` và `cpu_ms_per_img` (tổng thời gian tính của riêng từng khối đặc trưng).
  - `val_results.csv` (bước 2): `n_grid`, `grid_search_s`, `fit_best_s`, `time_s`.
  - `test_results.csv` (bước 3): `dim`, `fit_s` (train trên train+val), `predict_s`, `predict_ms_per_img`.
  - `run_info.json`: `total_wall_s` của mỗi bước.

> Nếu trong `data/` thiếu ảnh so với `labels.csv`, pipeline in cảnh báo và chỉ dùng ảnh đang có.

## Phiên bản kết quả

Tra cứu nhanh: mỗi dòng là một thư mục; chi tiết lệnh và tham số nằm trong `run_info.json` của thư mục đó. Từ 24/10/2026, `features/` và `results/` được chuyển vào `data/features/` và `data/results/` (chỉ đổi chỗ, nội dung giữ nguyên). Cột "Thực nghiệm" trỏ tới mục tương ứng trong [`EXPERIMENTS.md`](EXPERIMENTS.md).

**Đặc trưng (`data/features/`)**

| Phiên bản | Ngày | Số ảnh (train/val/test) | Nội dung | Thực nghiệm | Ghi chú |
|---|---|---|---|---|---|
| `f01` | 12/10/2026 | 261 / 42 / 216 | 10 khối: hog, lbp, color, sift, geom, sift_spm, hog_upper, sift_upper, gist, skin | 3 | Tạo bằng `run_all.sh`; có `timing_extract.csv` |
| `f02` | 24/10/2026 | 9.069 / 1.878 / 1.886 | Bộ gộp 4 nguồn, `labels_split_unk3000.csv`; trích đủ 10 khối như `f01` (r02 chỉ dùng `sift`, `hog`) | 5 | 247 s. `LABELS=data/train_test_split/labels_split_unk3000.csv DATA_DIR=data/dataset_6class_crops` |
| `f02_interrupted` | 24/10/2026 | 9.069 / 1.878 / 1.886 | Dở dang: chỉ có meta, geom và các khối ảnh (hog, lbp, color, hog_upper, gist, skin); chưa có SIFT, chưa có `run_info.json` | – | Lần chạy `LABELS=…labels_split_unk3000.csv FEATS="sift hog" MODELS=logreg bash run_all.sh f02 r02` bị dừng ở bước 1 để chuyển file split sang `data/train_test_split/`. Log: `run_log.txt`. **Không dùng** |

**Kết quả (`data/results/`)**

| Phiên bản | Ngày | Đặc trưng | Số ảnh (train/val/test) | Nội dung | Kết quả chính | Thực nghiệm | Ghi chú |
|---|---|---|---|---|---|---|---|
| `r01` | 12/10/2026 | `f01` | 261 / 42 / 216 | 12 bộ đặc trưng × 4 mô hình, test 15 cấu hình, có đo thời gian | val tốt nhất: sift + XGBoost 0.352; test tốt nhất: geom + XGBoost 0.283 | 1–4 | Cấu hình đã gồm mọi thay đổi của thực nghiệm 1–4 |
| `r02_interrupted` | 24/10/2026 | `f02` | 9.069 / 1.878 / 1.886 | Dở dang: bước 2 chỉ xong sift + LogReg (val 0.4298) | – | – | Bị dừng khi đang chạy hog + LogReg (lỗi `TerminatedWorkerError` trong `run_log.txt` là do dừng tay). **Không dùng** |
| **`r02`** | 24/10/2026 | `f02` | 9.069 / 1.878 / 1.886 | Bộ gộp, unknown 3000; `sift`, `hog` × LogReg, RF (RF bổ sung bằng `--update`) | val tốt nhất: sift + LogReg 0.430; test: sift + LogReg **0.434**, sift + RF 0.432, hog + LogReg 0.387 | 5, 6 | **Bản mới nhất** (đang hiển thị trong notebook). Bước 3 chạy lại sau khi thêm RF; kết quả LogReg không đổi |

Các lần chạy trong thực nghiệm 1–4 không lưu lại file; số liệu của chúng chỉ còn trong [`EXPERIMENTS.md`](EXPERIMENTS.md). `r01` tái lập đúng kết quả cuối của thực nghiệm 4.
Phiên bản tiếp theo nên đặt là `r03` (và `f03` nếu đổi dữ liệu hoặc tham số đặc trưng).

## Kết quả `r02` (bộ gộp 4 nguồn, unknown 3000, 24/10/2026)

Chạy trên **12.833 ảnh** (`data/train_test_split/labels_split_unk3000.csv`: train 9.069, val 1.878, test 1.886), 2 đặc trưng (`sift`, `hog`) × 2 mô hình (LogReg, RF). Biểu đồ đầy đủ trong `code/04_report.ipynb`.
Test có `unknown` đã giảm (23% thay vì 56%), nên precision các lớp hành động cao hơn so với phân bố thật; macro-F1 vẫn so được giữa các cấu hình trong `r02`, nhưng **không so trực tiếp được với `r01`** (khác dữ liệu, khác test).

| Đặc trưng | Mô hình | Siêu tham số tốt nhất | Val macro-F1 | Test macro-F1 | Test balanced acc |
|---|---|---|---|---|---|
| sift (500-d) | LogReg | C = 0.01 | **0.430** | **0.434** | **0.451** |
| sift (500-d) | RF | 300 cây, depth None, leaf 3 | 0.415 | 0.432 | 0.437 |
| hog (3780-d) | LogReg | C = 0.01 | 0.362 | 0.387 | 0.392 |
| hog (3780-d) | RF | 300 cây, depth None, leaf 3 | 0.339 | – | – |

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
- C tốt nhất là **0.01, giá trị nhỏ nhất của lưới** ở cả 2 đặc trưng: có thể C nhỏ hơn còn tốt hơn (xem Cần làm).
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

## Kết quả chạy thử (mẫu nhỏ, 12/10/2026)

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

## Thời gian chạy

Máy: Apple Silicon (arm64), 10 CPU (4 nhân hiệu năng cao + 6 nhân tiết kiệm điện), RAM 24 GB, `n_jobs=-1`.

### `f02` / `r02` (12.833 ảnh, 24/10/2026)

**Bước 1** (`f02`, 10 khối, `data/features/f02/timing_extract.csv`): tổng thời gian thực **247.2 s**.

| Công đoạn | Thời gian thực (s) |
|---|---|
| Lấy mẫu descriptor + KMeans K=500 (chỉ train) | 83.6 |
| Dense SIFT + BoVW (cả 3 split) | 130.5 |
| Đặc trưng ảnh hog/lbp/color/hog_upper/gist/skin (cả 3 split) | 32.2 |
| geom | 0.2 |

- CPU theo khối (cả 12.833 ảnh): sift_descriptor 842 s (65.6 ms/ảnh), bovw_encode 392 s, gist 136 s, lbp 82 s, hog 28 s (2.2 ms/ảnh).
- So với ước tính trước khi chạy (2–4 phút): thực tế 4 phút 7 s, vượt mức trên. Dense SIFT tốn ~66 ms/ảnh và bovw_encode ~31 ms/ảnh, gấp ~3.5 và ~11 lần `f01` (18.3 và 2.8 ms). **Nguyên nhân chưa xác định.** Không phải do kích thước crop: `sift_desc_pos` luôn resize về cạnh dài 256 px nên số điểm SIFT gần như không đổi. Giả thuyết (chưa kiểm chứng): `cpu_ms_per_img` đo bằng `perf_counter` trong từng tiến trình con, nên khi 10 tiến trình chạy lâu trên 4 nhân nhanh + 6 nhân chậm, cộng với `kmeans.predict` có thể tự mở thêm luồng trong mỗi tiến trình (oversubscription), thời gian mỗi ảnh bị đội lên; `f01` chỉ có 519 ảnh nên ít tranh chấp hơn.
- Các khối `r02` không dùng (lbp, color, gist, skin, hog_upper, geom) chỉ tốn ~25–30 s thời gian thực; KMeans và SIFT chiếm ~87% bước 1.

**Bước 2** (`r02`, 2 cấu hình × 4 giá trị C): tổng **16.3 s**. sift + LogReg: grid 1.5 s, fit lại 0.13 s; hog + LogReg: grid 11.6 s, fit lại 2.9 s.

**Bước 3** (2 cấu hình): tổng **4.4 s**. Fit trên train+val: sift 0.15 s, hog 3.9 s; dự đoán ≤ 0.03 ms/ảnh.

**Bước 2–4 bằng `run_all.sh`** (dùng lại `f02`, đo bằng `time`): **1 phút 14 s**; notebook khoảng 53 s (ước tính = 74 s trừ bước 2–3, gồm cả RF tính importance trên sift + hog).

**Bổ sung RF vào `r02`** (`02_train_select.py --update --feats sift hog --models rf`, rồi `03_final_test.py --update` và notebook; đo bằng `time`): **5 phút 24 s**.
- Bước 2: **262.1 s**. sift + RF: grid 22.8 s + fit lại 1.7 s; hog + RF: grid **216.5 s** + fit lại 20.9 s.
- Bước 3 (3 cấu hình: sift + LogReg, hog + LogReg, sift + RF): 7.0 s; fit sift + RF trên train+val 2.3 s.
- Notebook khoảng 55 s (ước tính = 324 s trừ bước 2–3).
- So với ước tính trước khi chạy (5–7 phút; hog + RF grid 2.5–4.5 phút): khớp. Ước tính dựa trên đo 20 cây trên 1 nhân (sift ~0.03 s/cây, hog ~0.28 s/cây).

### `f01` / `r01` (mẫu 519 ảnh, 12/10/2026)

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

## Thực nghiệm

Chỉ ghi **thực nghiệm mới nhất**. Toàn bộ nhật ký (mục 1–6), kết luận tạm và [bài học kinh nghiệm](EXPERIMENTS.md#bài-học-kinh-nghiệm) nằm trong [`EXPERIMENTS.md`](EXPERIMENTS.md).

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

## Cần làm

**Trạng thái hiện tại (để tiếp tục lần sau):**
- `.venv` đã cài đủ thư viện, `libomp` đã cài.
- `data/features/f02/` có đủ 10 khối đặc trưng cho bộ gộp (12.833 ảnh, `labels_split_unk3000.csv`). **Không cần chạy lại bước 1** khi chỉ đổi mô hình hoặc lưới: chạy `02_train_select.py --feat_version f02 --version r03 ...`.
- `data/results/r02/` là **kết quả mới nhất** (sift, hog × LogReg, RF); notebook đang hiển thị `r02`.
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

Cải thiện tiếp từ `r02`:
- [ ] **Mở rộng lưới C của LogReg về phía nhỏ** (thêm 0.001, 0.003): ở `r02`, C = 0.01 là giá trị nhỏ nhất của lưới và thắng ở cả sift và hog. Rẻ: bước 2 chỉ ~16 s.
- [ ] Cải thiện lớp `hold` (recall test 0.24–0.29 với LogReg, thấp nhất). Đã xem confusion (mục Đánh giá từng lớp): `hold` bị nhầm đều sang `eat/drink`, `talk_on_phone`, `work_on_computer`, và 48% lỗi nằm trong 4 lớp "tay cầm vật". Hướng thử: đặc trưng giữ vị trí vật so với đầu/tay (SIFT spatial pyramid, `upper` thân trên) thay cho BoVW toàn ảnh. Lý do: lỗi đến từ việc không phân biệt được vật ở đâu, không phải do thiếu dữ liệu.
- [ ] Xem bằng mắt các ảnh `eat/drink` → `work_on_computer` và `unknown` → `ride` để kiểm chứng giả thuyết "do bối cảnh" (bàn ăn/bàn làm việc, người đứng cạnh xe).
- [ ] **Late fusion sift + LogReg và sift + RF** (trung bình xác suất): 2 mô hình có macro-F1 gần bằng nhau (0.434 / 0.432) nhưng mạnh ở các lớp khác nhau (LogReg: `eat/drink`, `talk_on_phone`; RF: `hold`, `ride`, `unknown`). Chọn trọng số trên val. Rẻ: cả 2 mô hình đã lưu trong `data/results/r02/models/`, nhưng các model đó train trên train+val, nên chọn trọng số phải fit lại trên train.

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

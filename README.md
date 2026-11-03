# Human-Object Interaction Recognition (V-COCO, 6 lớp)

Đồ án môn Thị giác máy tính: nhận diện hành động người–vật. Bài toán tham khảo từ DP-HOI ([Li et al., CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Disentangled_Pre-training_for_Human-Object_Interaction_Detection_CVPR_2024_paper.pdf)). Bài báo làm **phát hiện** HOI (định vị người, vật và hành động); đồ án này chỉ làm nhánh **nhận diện hành động** trên ảnh crop người, và dùng phương pháp khác với bài báo. Tóm tắt input, output, metric của bài báo và so sánh với đồ án: [`doc/README_paper.md`](doc/README_paper.md).

Phân loại hành động của 1 người trong ảnh crop thành 6 lớp `hold, ride, eat/drink, talk_on_phone, work_on_computer, unknown`
bằng **đặc trưng thủ công** + **học máy cổ điển**.

- 4 đặc trưng cơ bản: HOG (dáng người), LBP (kết cấu), Color-HSV (màu/bối cảnh), SIFT-BoVW (chi tiết cục bộ).

  **Quy ước tên** (`config.FEATURE_NAMES`, `config.MODEL_NAMES`): bảng, hình, nhận xét và slide dùng **tên hiển thị**; tên trong code chỉ xuất hiện ở tên file, lệnh chạy và đoạn code. Không đổi khoá trong code vì mọi file đặc trưng / kết quả (`f01`–`f03`, `r01`–`r05`) đặt theo khoá này.

  | Tên hiển thị | Tên trong code | Tên đầy đủ | Cách tính (đúng như `code/features.py`) | Số chiều |
  |---|---|---|---|---|
  | HOG | `hog` | Histogram of Oriented Gradients | ảnh xám resize 64×128; 9 hướng, ô 8×8, khối 2×2, chuẩn hoá L2-Hys (skimage) | 3,780 |
  | LBP | `lbp` | LBP histogram (uniform, bất biến xoay) | ảnh xám resize 128×256; LBP `uniform` P = 24, R = 3 (skimage: 26 mẫu uniform bất biến xoay); histogram 26 bin trên mỗi ô của lưới 3×3, chuẩn hoá L1 từng ô | 234 |
  | Color-HSV | `color` | Color histogram HSV | ảnh gốc (không resize) đổi sang HSV; histogram 3 chiều 8×8×8 cho nửa trên và nửa dưới, chuẩn hoá L1 | 1,024 |
  | SIFT-BoVW | `sift` | Dense SIFT + Bag of Visual Words | ① dense SIFT (OpenCV) trên ảnh xám resize cạnh dài 256, lưới bước 8, cỡ 16: mỗi điểm 1 descriptor 128 chiều; ② codebook MiniBatchKMeans K = 500 học trên ~200,000 descriptor lấy mẫu từ train; ③ gán mỗi descriptor vào từ gần nhất, đếm thành histogram, chuẩn hoá L1 rồi căn bậc 2 (Hellinger) | 500 |

  Mốc so sánh: **CLIP** (`clip`, embedding ảnh CLIP ViT-B/32, 512 chiều). Mô hình: **LogReg** (`logreg`), **SVM** (`svm`, kernel RBF), **RF** (`rf`), **XGBoost** (`xgb`), **zero-shot** (`clip_zs`). Cấu hình viết là "đặc trưng + mô hình", ví dụ SIFT-BoVW + SVM.

  **Cách ghi số** (theo bài báo CV): chỉ số (macro-F1, accuracy, precision, recall, balanced accuracy) ghi theo **%, 1 chữ số thập phân** (ví dụ macro-F1 46.4, accuracy 48.8); chênh lệch giữa hai chỉ số tính bằng điểm % (+42.5). Dấu thập phân là `.`, phân cách hàng nghìn là `,` (12,833 ảnh; RAM 4,008 MB). Ngưỡng T, siêu tham số, tương quan Spearman, thời gian, dung lượng giữ đơn vị gốc. File csv và bảng `pandas` trong notebook lưu giá trị gốc 0–1; notebook hiển thị theo %.
- Đặc trưng bổ sung, thiết kế theo đặc điểm dữ liệu (vật nhỏ gần đầu, vật bị cắt khỏi crop, tỉ lệ khung người). **Chỉ mới chạy ở `r01` (mẫu 519 ảnh)**; kết quả chính (`r03`, `r05`) trên bộ gộp chỉ dùng 4 đặc trưng cơ bản. Các khối này vẫn được trích trong `f02`/`f03` nhưng chưa train:

  | Tên | Mô tả | Số chiều |
  |---|---|---|
  | `geom` | hình học bbox người: log tỉ lệ w/h, log w, log h, log diện tích, mức nới thực tế 4 phía (người sát mép ảnh) | 8 |
  | `sift_spm` | SIFT-BoVW + spatial pyramid (1x1 + 2x2), giữ vị trí của "từ" | 2,500 |
  | `upper` | HOG (1,764) + SIFT-BoVW (500) chỉ trên 40% phía trên crop: đầu, tay gần mặt | 2,264 |
  | `gist` | năng lượng Gabor 4 tỉ lệ × 8 hướng trên lưới 4x4: bối cảnh | 512 |
  | `skin` | tỉ lệ da (YCrCb) trên lưới 4x4 + diện tích/trọng tâm/độ trải | 21 |

- Phối hợp (cũng chỉ mới chạy ở `r01`): `fused` (4 cơ bản), `combo_local` (sift_spm + sift_upper + color + geom + skin), `fused_all` (tất cả). Danh sách bộ đặc trưng nằm trong `FEATURE_SETS` ở `code/models.py`: thêm tổ hợp mới chỉ cần thêm 1 dòng.
- 4 mô hình: **Logistic Regression**, **SVM (RBF)**, **Random Forest**, **XGBoost**. Cân bằng lớp (vì `unknown` chiếm ~65%) bằng `class_weight`; riêng XGBoost dùng `sample_weight`. SVM và XGBoost giảm chiều bằng PCA 256 khi bộ đặc trưng > 1,500 chiều.
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
├── data/models/                 (tự sinh) clip_svm_proba_r05.joblib: CLIP + SVM r05 bật probability, do predict_clip_svm.py --proba tạo
├── code/
│   ├── config.py                đường dẫn, danh sách lớp, tham số đặc trưng
│   ├── data_utils.py            đọc labels.csv, đọc ảnh (bỏ qua ảnh không có trên đĩa)
│   ├── features.py              hog, lbp, color, sift (= SIFT-BoVW; + sift_spm, sift_upper), hog_upper, gist, skin, geom
│   ├── models.py                4 mô hình + lưới siêu tham số + FEATURE_SETS (cách phối)
│   ├── versioning.py            quản lý phiên bản, chặn ghi đè, ghi run_info.json
│   ├── resources.py             đo RAM đỉnh, CPU, dung lượng, thông tin máy
│   ├── clip_feat.py             (tuỳ chọn) embedding ảnh/prompt CLIP ViT-B/32: mốc so sánh, không phải đặc trưng thủ công
│   ├── 01_extract_features.py   trích đặc trưng -> data/features/<fXX>/
│   ├── 02_train_select.py       grid search trên val (các bộ đặc trưng trong FEATURE_SETS x 4 mô hình) -> data/results/<rXX>/val_results.csv
│   ├── 03_final_test.py         train lại trên train+val, đánh giá test -> data/results/<rXX>/
│   ├── 04_report.ipynb          báo cáo chính: hiển thị r05 (mục 1–7: 4 đặc trưng thủ công; mục 8: CLIP)
│   ├── predict_clip_svm.py      dự đoán ảnh mới bằng CLIP + SVM r05 (--proba: độ tin cậy, top-3)
│   └── fig_clip_svm_proba.py    vẽ Hình 8 của báo cáo (CLIP + SVM, top-3 xác suất) -> doc/hinh8_clip_svm_top3.png
├── doc/                         README_paper.md (tóm tắt input, output, metric của DP-HOI); PDF bài báo chỉ để ở máy, không có trên repo
├── run_all.sh                   chạy tuần tự bước 1 → 4
└── requirements.txt
```

`data/features/`, `data/results/` và `data/models/` do script sinh ra, không có trên repo.

## Dữ liệu

Trong `data/` hiện có 2 bộ:

- `data/vcoco_6class_crops/`: bộ **pipeline đang dùng** (`DATA_DIR` trong `code/config.py`). Chỉ gồm V-COCO, đã chia train/val/test. Trên đĩa mới có 519/15,187 ảnh.
- `data/dataset_6class_crops/`: bộ **mới, gộp 4 nguồn** (V-COCO, PASCAL VOC 2012 Action, HICO-DET, Stanford40), đã lọc tay. Pipeline **chưa dùng**. Mô tả chi tiết và quy tắc gán nhãn ở `data/dataset_6class_crops/README.md`.

**Số lượng `dataset_6class_crops`** (kiểm tra ngày 24/10/2026): có 22,366 file `.jpg`, khớp 22,366 dòng trong `labels.csv`, không thiếu file, không trùng đường dẫn.

| Lớp | v_coco | pascal_voc | hico_det | stanford40 | Tổng | Tỉ lệ |
|---|---:|---:|---:|---:|---:|---:|
| hold | 1,416 | 0 | 423 | 862 | 2,701 | 12.1% |
| ride | 853 | 951 | 293 | 767 | 2,864 | 12.8% |
| eat/drink | 1,145 | 0 | 86 | 255 | 1,486 | 6.6% |
| talk_on_phone | 615 | 436 | 20 | 256 | 1,327 | 5.9% |
| work_on_computer | 777 | 427 | 27 | 224 | 1,455 | 6.5% |
| unknown | 9,284 | 939 | 571 | 1,739 | 12,533 | 56.0% |
| **Tổng** | 14,090 | 2,753 | 1,420 | 4,103 | **22,366** | |

- **Không có cột `split`**, chỉ có cột `split_goc` (split của nguồn gốc): v_coco gồm train 6,340, val 1,046, test 6,704; pascal_voc 2,753 ảnh trainval; hico_det 1,420 ảnh train; stanford40 4,103 ảnh để trống.
- `unknown` chiếm 56% toàn bộ, riêng v_coco là 66%. pascal_voc không có `hold` và `eat/drink`.
- `labels_ignore.csv` ghi 10,002 người bị loại (không xuất crop): stanford40 5,388, pascal_voc 3,214, hico_det 1,400. Lý do nhiều nhất là "tay cầm vật ngoài danh sách vật cầm tay" (3,058).
- Thư mục lớp là `eat_drink`, nhưng trong `labels.csv` nhãn vẫn ghi `eat/drink`.

**Chia train/val/test** (24/10/2026): `code/train_test_split.py`, seed 2026, tỉ lệ 70/15/15. Kết quả ghi ra `data/train_test_split/labels_split.csv` (= `labels.csv` + cột `split`); `labels.csv` giữ nguyên. `crop_path` vẫn tính từ `data/dataset_6class_crops/`, nên khi chạy pipeline phải chỉ cả thư mục ảnh: `LABELS=data/train_test_split/labels_split_unk3000.csv DATA_DIR=data/dataset_6class_crops bash run_all.sh ...` (hoặc `01_extract_features.py --labels ... --data_dir ...`).
- Chia **theo ảnh gốc** (`dataset` + `source_image`, 17,304 ảnh): mọi crop của cùng một ảnh nằm chung 1 split. Phân tầng theo lớp × nguồn bằng `StratifiedGroupKFold` 20 fold (14 fold train, 3 val, 3 test).
- **Không** giữ split gốc (`split_goc`) của V-COCO: test mới lấy ảnh từ cả 4 nguồn, nên kết quả trên test này không so trực tiếp được với test V-COCO.
- File `labels_split.csv` giữ đủ 12,533 `unknown`.
- **Bản giảm `unknown`**: `train_test_split.py --max_unknown 3000` → `data/train_test_split/labels_split_unk3000.csv` (12,833 crop). Sau khi chia, các dòng `unknown` được xáo theo seed 2026 và chỉ giữ 3,000 dòng đầu, tính trên cả 3 split. Split của mỗi crop giữ nguyên như trong `labels_split.csv`. Còn lại: train 9,069 / val 1,878 / test 1,886. `unknown` còn 2,087 / 456 / 457, tức 23% thay vì 56%. Vì giảm cả ở test, precision các lớp hành động trên test sẽ cao hơn so với khi giữ đủ `unknown`; khi báo cáo cần ghi rõ.

| Lớp | train | val | test | Tổng |
|---|---:|---:|---:|---:|
| hold | 1,916 | 384 | 401 | 2,701 |
| ride | 2,053 | 422 | 389 | 2,864 |
| eat/drink | 1,078 | 200 | 208 | 1,486 |
| talk_on_phone | 920 | 192 | 215 | 1,327 |
| work_on_computer | 1,015 | 224 | 216 | 1,455 |
| unknown | 8,686 | 1,944 | 1,903 | 12,533 |
| **Tổng** | **15,668 (70.1%)** | **3,366 (15.0%)** | **3,332 (14.9%)** | **22,366** |

Theo nguồn (train / val / test): v_coco 9,866 / 2,124 / 2,100; stanford40 2,869 / 619 / 615; pascal_voc 1,938 / 412 / 403; hico_det 995 / 211 / 214.

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

Biến môi trường tuỳ chọn của `run_all.sh`: `LABELS` (file nhãn có cột `split`), `DATA_DIR` (thư mục ảnh mà `crop_path` tính từ đó; mặc định cạnh file nhãn), `FEATS`, `MODELS` (mặc định: mọi bộ đặc trưng có trong phiên bản đặc trưng × mọi mô hình), `CLIP=1` (bước 1 trích thêm khối CLIP). `LABELS`/`DATA_DIR` chỉ dùng ở bước 1 và được ghi vào `data/features/<fXX>/run_info.json`, nên notebook tự đọc đúng dữ liệu của từng phiên bản.

Chỉ chạy lại notebook `04_report.ipynb` thì **không** tạo ra kết quả: notebook chỉ đọc `data/features/` và `data/results/` do bước 1–3 sinh ra. Repo không chứa hai thư mục này, nên lần đầu phải chạy `run_all.sh` (hoặc bước 1 → 3 bên dưới) trước. Với cùng dữ liệu và thư viện trong `requirements.txt`, kết quả tái lập chính xác (đã kiểm tra: đặc trưng giống từng bit, val/test lệch 0).

Hoặc chạy từng bước trong thư mục `code/` (ví dụ tạo `f02` và `r02`):

```bash
cd code
../.venv/bin/python 01_extract_features.py --version f02 --note "mô tả"                 # bỏ qua nếu dùng lại đặc trưng cũ
../.venv/bin/python 02_train_select.py --feat_version f02 --version r02 --note "mô tả"  # chọn mô hình trên val
../.venv/bin/python 03_final_test.py --version r02                                      # đánh giá test 1 lần
RUN_VERSION=r02 ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 04_report.ipynb  # báo cáo hiển thị r02
```

**Báo cáo kết quả xem ở `code/04_report.ipynb`**: notebook hiển thị phiên bản mới nhất (hiện là `r05`): mục 1–7 chỉ có 4 đặc trưng thủ công (kết quả chính), mục 8 tách riêng CLIP (mốc so sánh). Khi có phiên bản mới, sửa giá trị mặc định `VERSION` ở ô đầu rồi Run All (kernel `.venv`). Muốn xem lại phiên bản cũ thì đặt `VERSION` cũ và chạy lại; `data/results/` chỉ chứa kết quả do script sinh ra, không chứa bản sao notebook hay HTML.

Tùy chọn:
- Thử nhanh: `01_extract_features.py --version f_test --limit 60` (lấy khoảng 10 ảnh mỗi lớp mỗi split).
- Chạy thêm một phần vào phiên bản có sẵn: `02_train_select.py --feat_version f02 --version r02 --update --feats sift color --models svm` (chỉ thay các dòng `--feats × --models`, giữ nguyên các dòng khác). Sau đó chạy `03_final_test.py --version r02 --update`.
- Đổi tham số đặc trưng trong `config.py`, sau đó chạy bước 1 với **phiên bản mới**.
- **CLIP (mốc so sánh)**: `01_extract_features.py --clip` trích thêm khối CLIP (embedding ảnh 512 chiều, ảnh crop được đệm thành hình vuông) và `clip_text.npy` (embedding 5 prompt trong `config.CLIP_PROMPTS`). Bộ đặc trưng CLIP chạy được với mọi mô hình (LogReg = linear probe; SVM RBF, RF, XGBoost là bộ phân loại phi tuyến) và với **`clip_zs`** (zero-shot): softmax(100 × cosine ảnh–prompt) trên 5 lớp hành động, lấy lớp cao nhất; nếu xác suất cao nhất < T thì gán `unknown`. T ∈ {0, 0.5, 0.6, 0.7, 0.8, 0.9} chọn trên val. Ngưỡng này mô phỏng lỏng ngưỡng T chọn Reliable Person Query của DP-HOI (ở bài báo là ngưỡng điểm phát hiện người, ở đây là ngưỡng tin cậy hành động). Kết quả CLIP thuộc nhóm `group = clip`, chọn cấu hình tốt nhất riêng với nhóm đặc trưng thủ công. **Lưu ý dữ liệu**: CLIP train trên ~400 triệu cặp ảnh–văn bản lấy từ web không công bố, không loại trừ được ảnh COCO (có ảnh test V-COCO); chỉ báo cáo như mốc so sánh.
- **Dự đoán ảnh mới bằng CLIP + SVM**: `cd code && ../.venv/bin/python predict_clip_svm.py anh.jpg thu_muc_anh/ [--proba] [--out ket_qua.csv]`. Script nạp `data/results/r05/models/clip_svm.joblib`, tính embedding CLIP đúng như lúc train (đệm ảnh thành hình vuông, ViT-B/32) rồi in nhãn dự đoán. Đầu vào nên là ảnh cắt quanh người như `data/dataset_6class_crops`. Thêm `--proba` thì in độ tin cậy và top-3 lớp, dùng bản SVM `probability=True` lưu ở `data/models/clip_svm_proba_r05.joblib`; lần đầu chưa có file này thì script tự fit (~1 phút) rồi lưu. Nhãn ở cả hai chế độ đều trùng `r05`. Đã kiểm tra 03/11/2026: embedding sinh lại trùng từng bit với `f03`, kết quả 12 ảnh khớp Hình 8; mỗi lần chạy mất ~4 s cho vài ảnh (chủ yếu là nạp CLIP). Cảnh báo `QuickGELU mismatch` của open_clip là bình thường, lúc trích `f03` cũng dùng đúng cấu hình này.
- **Hình 8 của báo cáo (CLIP + SVM kèm xác suất)**: `cd code && ../.venv/bin/python fig_clip_svm_proba.py` → `doc/hinh8_clip_svm_top3.png`. Mô hình `clip_svm.joblib` ở `r05` không bật `probability`, nên script fit lại đúng cấu hình đó (StandardScaler + SVC RBF, C = 10, γ = scale, `class_weight=balanced`) với `probability=True` (Platt scaling, CV 5-fold nội bộ) trên train+val, rồi vẽ 12 ảnh test của Hình 7 kèm top-3 xác suất. Chỉ đọc `data/`, không ghi vào `data/results/`. Lần chạy 03/11/2026 (1,886 ảnh test): test macro-F1 88.9 (bằng `r05`), nhãn dự đoán lệch so với `r05` 0 / 1,886 ảnh, tức bật xác suất không làm đổi kết quả; fit 38.0 s, cả script 45.3 s. Trên 12 ảnh: đúng 10/12, độ tin cậy 84–99%; 2 ảnh sai vẫn tự tin (ride → unknown 87%, unknown → eat/drink 94%).

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
| `f02` | 24/10/2026 | 9,069 / 1,878 / 1,886 | Bộ gộp 4 nguồn, `labels_split_unk3000.csv`; trích đủ 10 khối như `f01` (r02 chỉ dùng SIFT-BoVW, HOG) | 5 | 247 s. `LABELS=data/train_test_split/labels_split_unk3000.csv DATA_DIR=data/dataset_6class_crops` |
| `f02_interrupted` | 24/10/2026 | 9,069 / 1,878 / 1,886 | Dở dang: chỉ có meta, geom và các khối ảnh (hog, lbp, color, hog_upper, gist, skin); chưa có SIFT, chưa có `run_info.json` | – | Lần chạy `LABELS=…labels_split_unk3000.csv FEATS="sift hog" MODELS=logreg bash run_all.sh f02 r02` bị dừng ở bước 1 để chuyển file split sang `data/train_test_split/`. Log: `run_log.txt`. **Không dùng** |
| `f03` | 24/10/2026 | 9,069 / 1,878 / 1,886 | Như `f02` + khối CLIP (CLIP ViT-B/32, 512 chiều, ảnh đệm vuông) và `clip_text.npy` (5 prompt); khối thủ công giống từng bit `f02` | 8 | 400.6 s; RAM đỉnh 4.0 GB. `LABELS=data/train_test_split/labels_split_unk3000.csv DATA_DIR=data/dataset_6class_crops CLIP=1` |

**Kết quả (`data/results/`)**

| Phiên bản | Ngày | Đặc trưng | Số ảnh (train/val/test) | Nội dung | Kết quả chính | Thực nghiệm | Ghi chú |
|---|---|---|---|---|---|---|---|
| `r01` | 12/10/2026 | `f01` | 261 / 42 / 216 | 12 bộ đặc trưng × 4 mô hình, test 15 cấu hình, có đo thời gian | val tốt nhất: SIFT-BoVW + XGBoost 35.2; test tốt nhất: geom + XGBoost 28.3 | 1–4 | Cấu hình đã gồm mọi thay đổi của thực nghiệm 1–4 |
| `r02_interrupted` | 24/10/2026 | `f02` | 9,069 / 1,878 / 1,886 | Dở dang: bước 2 chỉ xong SIFT-BoVW + LogReg (val 43.0) | – | – | Bị dừng khi đang chạy HOG + LogReg (lỗi `TerminatedWorkerError` trong `run_log.txt` là do dừng tay). **Không dùng** |
| `r02` | 24/10/2026 | `f02` | 9,069 / 1,878 / 1,886 | Bộ gộp, unknown 3,000; SIFT-BoVW, HOG × LogReg, RF (RF bổ sung bằng `--update`) | val tốt nhất: SIFT-BoVW + LogReg 43.0; test: SIFT-BoVW + LogReg **43.4**, SIFT-BoVW + RF 43.2, HOG + LogReg 38.7 | 5, 6 | Bước 3 chạy lại sau khi thêm RF; kết quả LogReg không đổi |
| `r03` | 24/10/2026 | `f02` | 9,069 / 1,878 / 1,886 | HOG, LBP, Color-HSV, SIFT-BoVW × LogReg, SVM, RF, XGBoost; test cả 16 cấu hình; đủ chỉ số (accuracy, precision, F1, confusion số ảnh); đo RAM | val tốt nhất: SIFT-BoVW + SVM 48.0, test **46.4**; test cao nhất: HOG + SVM 48.3 (val 44.5) | 7 | 26 phút 56 s; RAM đỉnh 7.8 GB. Bảng đầy đủ 4 đặc trưng thủ công × 4 mô hình |
| `r04` | 24/10/2026 | `f03` | 9,069 / 1,878 / 1,886 | CLIP × LogReg, SVM, RF, XGBoost, zero-shot (`clip_zs`, lưới T); SIFT-BoVW × 4 mô hình làm mốc; test cả 9 cấu hình | nhóm CLIP: CLIP + SVM val 87.8, test **88.9**; zero-shot T = 0.6, test 58.1; nhóm thủ công: SIFT-BoVW + SVM test 46.4 | 8 | CLIP là mốc so sánh (rủi ro dữ liệu web). 15 phút 37 s cả pipeline |
| **`r05`** | 31/10/2026 | `f03` | 9,069 / 1,878 / 1,886 | HOG, LBP, Color-HSV, SIFT-BoVW × LogReg, SVM, RF, XGBoost (kết quả chính) + CLIP × LogReg, SVM, RF, XGBoost, `clip_zs` (mốc so sánh); test cả 21 cấu hình | Trùng chính xác `r03` + `r04`: thủ công SIFT-BoVW + SVM val 48.0, test **46.4**; CLIP + SVM test 88.9; zero-shot 58.1 | – | **Bản mới nhất** (notebook hiển thị mặc định). Gộp `r03` + `r04` để 1 notebook hiện đủ; 31 phút 56 s |

Các lần chạy trong thực nghiệm 1–4 không lưu lại file; số liệu của chúng chỉ còn trong [`EXPERIMENTS.md`](EXPERIMENTS.md). `r01` tái lập đúng kết quả cuối của thực nghiệm 4.
Phiên bản tiếp theo nên đặt là `r06` (và `f04` nếu đổi dữ liệu, tham số đặc trưng hoặc thêm khối).

## Kết quả `r05` (4 đặc trưng thủ công + CLIP tách riêng, 31/10/2026)

README chỉ giữ kết quả mới nhất; kết quả và thời gian chạy của các phiên bản cũ (`r04`, `r03`, `r02`, `r01`) nằm trong [`EXPERIMENTS.md`](EXPERIMENTS.md).

Chạy trên **12,833 ảnh** (`f03`, `labels_split_unk3000.csv`: train 9,069, val 1,878, test 1,886): 4 đặc trưng thủ công (HOG, LBP, Color-HSV, SIFT-BoVW) × LogReg, SVM, RF, XGBoost (16 cấu hình, **kết quả chính**) và CLIP × LogReg, SVM, RF, XGBoost, zero-shot (5 cấu hình, **mốc so sánh, tách riêng**); test cả 21 cấu hình. `r05` gộp `r03` và `r04` vào một phiên bản để notebook hiển thị đủ; **mọi số val/test trùng chính xác `r03` và `r04`** (chênh 0 ở cả 21 cấu hình), nên các bảng dưới giữ nguyên số của hai phiên bản đó. Notebook `code/04_report.ipynb`: mục 1–7 chỉ 4 đặc trưng thủ công, mục 8 là CLIP.

### Kết quả chính: 4 đặc trưng thủ công × 4 mô hình

4 đặc trưng cơ bản (HOG, LBP, Color-HSV, SIFT-BoVW) × 4 mô hình (LogReg, SVM, RF, XGBoost), **test cả 16 cấu hình**. Cấu hình được chọn là cấu hình tốt nhất **trên val**: **SIFT-BoVW + SVM**. Số test của các cấu hình khác chỉ để báo cáo. Test có `unknown` đã giảm (23% thay vì 56%), nên precision các lớp hành động cao hơn so với phân bố thật. Biểu đồ đầy đủ trong `code/04_report.ipynb`.

Xếp theo val macro-F1. Accuracy chỉ để tham khảo; macro recall = balanced accuracy. RAM là RAM đỉnh khi grid search ở bước 2 (cận trên, gồm nền worker; xem mục Thời gian chạy). Cột thời gian và RAM bước 2 là số đo của `r05`.

| Đặc trưng | Mô hình | Siêu tham số tốt nhất | Val macro-F1 | Test macro-F1 | Test accuracy | Test macro precision | Test macro recall | Test weighted F1 | Bước 2 (s) | RAM bước 2 (MB) | Mô hình (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **SIFT-BoVW** | **SVM** | C = 1, γ = scale | **48.0** | 46.4 | 48.8 | 46.0 | 47.8 | 48.4 | 109.4 | 3,807 | 41.45 |
| Color-HSV | XGBoost | lr 0.1, depth 4, 600 cây | 45.8 | 43.2 | 48.1 | 43.1 | 43.8 | 47.2 | 196.5 | 4,573 | 5.81 |
| SIFT-BoVW | XGBoost | lr 0.1, depth 6, 300 cây | 44.9 | 44.7 | 47.7 | 44.8 | 45.2 | 47.2 | 60.7 | 3,882 | 5.50 |
| HOG | SVM | C = 10, γ = scale | 44.5 | **48.3** | **52.6** | **49.4** | **47.9** | **52.1** | 48.1 | 6,743 | 26.39 |
| SIFT-BoVW | LogReg | C = 0.01 | 43.0 | 43.4 | 45.7 | 43.2 | 45.1 | 45.5 | 0.9 | 2,520 | 0.04 |
| LBP | SVM | C = 1, γ = scale | 42.2 | 40.9 | 44.1 | 41.0 | 42.0 | 43.9 | 48.2 | 3,758 | 19.75 |
| SIFT-BoVW | RF | 300 cây, depth None, leaf 3 | 41.5 | 43.2 | 46.8 | 45.5 | 43.7 | 45.6 | 29.9 | 3,666 | 114.64 |
| LBP | XGBoost | lr 0.1, depth 6, 600 cây | 40.9 | 40.1 | 45.3 | 42.3 | 40.2 | 44.0 | 48.6 | 3,905 | 13.06 |
| LBP | LogReg | C = 0.1 | 40.1 | 36.6 | 39.7 | 36.1 | 38.6 | 38.6 | 0.8 | 2,156 | 0.02 |
| HOG | XGBoost | lr 0.05, depth 4, 600 cây | 39.9 | 44.4 | 47.9 | 44.3 | 45.4 | 47.0 | 96.6 | 5,845 | 9.95 |
| Color-HSV | RF | 300 cây, depth None, leaf 3 | 39.3 | 37.6 | 44.0 | 39.8 | 38.8 | 41.4 | 77.7 | 3,162 | 105.71 |
| Color-HSV | SVM | C = 1, γ = scale | 37.8 | 35.9 | 40.2 | 35.7 | 37.0 | 39.2 | 259.8 | 4,377 | 85.50 |
| LBP | RF | 300 cây, depth None, leaf 3 | 37.2 | 34.7 | 41.3 | 38.8 | 35.6 | 39.1 | 48.5 | 4,178 | 107.09 |
| HOG | LogReg | C = 0.01 | 36.2 | 38.7 | 41.1 | 38.7 | 39.2 | 41.5 | 15.3 | 4,562 | 0.27 |
| HOG | RF | 300 cây, depth None, leaf 3 | 33.9 | 35.3 | 43.1 | 44.5 | 35.9 | 40.3 | 250.6 | 5,965 | 98.42 |
| Color-HSV | LogReg | C = 0.01 | 33.5 | 32.7 | 35.5 | 32.7 | 33.3 | 35.7 | 4.5 | 3,880 | 0.07 |

Nhận xét:
- **SVM RBF giúp rõ với SIFT-BoVW**: SIFT-BoVW + SVM đạt test **46.4**, so với 43.4 của SIFT-BoVW + LogReg (cấu hình tốt nhất của `r02`); nhưng không phải mọi mô hình phi tuyến đều hơn (SIFT-BoVW + RF 43.2). SVM là mô hình tốt nhất trên val cho 3/4 đặc trưng (SIFT-BoVW, HOG, LBP); XGBoost tốt nhất cho Color-HSV.
- **Color-HSV mạnh nhất với XGBoost**: Color-HSV + LogReg 33.5 (kém nhất), Color-HSV + XGBoost 45.8 (thứ 2 trên val); RF 39.3, SVM 37.8. Histogram màu không tách tuyến tính được, nhưng XGBoost tìm được ngưỡng trên từng bin màu (bối cảnh: đường, cỏ, bàn).
- **RF kém hơn SVM và XGBoost ở 3/4 đặc trưng** (trừ Color-HSV: Color-HSV + RF 39.3 > Color-HSV + SVM 37.8; sửa 31/10/2026, trước ghi nhầm "ở cả 4 đặc trưng"), lại nặng nhất (98–115 MB mỗi mô hình) và chậm (HOG + RF 250.6 s ở bước 2 của `r05`).
- **Val dự đoán test tốt**: tương quan hạng Spearman val–test 0.86 trên 16 cấu hình; chênh lệch test − val trung bình −0.2 (từ −3.5 đến +4.6). Riêng **HOG + SVM** đứng thứ 4 trên val (44.5) nhưng cao nhất trên test (48.3). Không chọn lại theo test: chênh 3.5 trên val giữa SIFT-BoVW + SVM và HOG + SVM vẫn là căn cứ duy nhất được phép dùng.
- **Giá trị tốt nhất nằm ở biên lưới**: LogReg C = 0.01 (nhỏ nhất) ở HOG, Color-HSV, SIFT-BoVW; SVM C = 10 (lớn nhất) ở HOG; XGBoost 600 cây (lớn nhất) ở 3/4 đặc trưng. Lưới chưa đủ rộng (xem mục Cần làm).

#### Đánh giá từng lớp (test)

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

Ma trận nhầm lẫn (số ảnh) của **SIFT-BoVW + SVM** (hàng = nhãn thật, cột = dự đoán; `data/results/r05/confusion_sift_svm.csv`):

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
- **HOG + SVM mạnh ở lớp dáng người**: `hold` recall 46.6 (SIFT-BoVW + SVM 27.4), `ride` F1 73.1. HOG mã hoá dáng người (ngồi trên xe/ngựa, tay đưa ra cầm vật); SIFT-BoVW mạnh ở lớp có vật đặc trưng (`work_on_computer`, `talk_on_phone`). Hai đặc trưng bổ sung nhau, nên ghép HOG + SIFT-BoVW (hoặc late fusion 2 SVM) đáng thử (xem mục Cần làm).
- **`ride` ↔ `unknown`** là lỗi lớn thứ 2 sau nhóm "tay cầm vật", và tính theo cặp hai lớp thì là cặp nhầm qua lại nhiều nhất (72 `unknown` → `ride` + 54 `ride` → `unknown` = 126, sát `hold` ↔ `eat/drink` 121), như `r02`.

### Mốc so sánh: CLIP (tách riêng)

> **Lưu ý dữ liệu**: CLIP được train trên ~400 triệu cặp ảnh–văn bản lấy từ web, không công bố. Không loại trừ được việc dữ liệu đó có ảnh COCO, kể cả ảnh test V-COCO (trong test, 1,025/1,886 ảnh là từ v_coco, tức ảnh COCO). Vì vậy kết quả CLIP chỉ là **mốc so sánh**, không thay kết quả chính của đặc trưng thủ công.

Xếp theo nhóm, rồi theo val macro-F1. Cấu hình chọn theo val trong từng nhóm: **CLIP + SVM** và **SIFT-BoVW + SVM**. Accuracy chỉ để tham khảo; macro recall = balanced accuracy. RAM là RAM đỉnh khi grid search ở bước 2 (cận trên, gồm nền worker).

| Nhóm | Đặc trưng | Mô hình | Siêu tham số tốt nhất | Val macro-F1 | Test macro-F1 | Test accuracy | Test macro precision | Test macro recall | Test weighted F1 | Bước 2 (s) | RAM bước 2 (MB) | Mô hình (MB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CLIP | **CLIP** | **SVM** | C = 10, γ = scale | **87.8** | **88.9** | **89.2** | **89.3** | **88.5** | **89.2** | 60.0 | 3,253 | 23.90 |
| CLIP | CLIP | XGBoost | lr 0.1, depth 4, 600 cây | 87.0 | 88.2 | 88.5 | 88.7 | 87.9 | 88.6 | 154.3 | 3,177 | 5.68 |
| CLIP | CLIP | LogReg | C = 0.01 | 87.0 | 87.6 | 88.0 | 87.2 | 88.1 | 88.0 | 2.4 | 2,698 | 0.04 |
| CLIP | CLIP | RF | 300 cây, depth None, leaf 3 | 84.7 | 83.3 | 83.6 | 86.5 | 82.0 | 83.9 | 109.2 | 2,889 | 69.21 |
| CLIP | CLIP | **zero-shot** | T = 0.6 | 60.5 | 58.1 | 54.7 | 65.1 | 61.5 | 50.8 | 0.2 | 1,428 | 0.01 |
| thủ công | **SIFT-BoVW** | **SVM** | C = 1, γ = scale | 48.0 | 46.4 | 48.8 | 46.0 | 47.8 | 48.4 | 109.4 | 3,807 | 41.45 |
| thủ công | SIFT-BoVW | XGBoost | lr 0.1, depth 6, 300 cây | 44.9 | 44.7 | 47.7 | 44.8 | 45.2 | 47.2 | 60.7 | 3,882 | 5.50 |
| thủ công | SIFT-BoVW | LogReg | C = 0.01 | 43.0 | 43.4 | 45.7 | 43.2 | 45.1 | 45.5 | 0.9 | 2,520 | 0.04 |
| thủ công | SIFT-BoVW | RF | 300 cây, depth None, leaf 3 | 41.5 | 43.2 | 46.8 | 45.5 | 43.7 | 45.6 | 29.9 | 3,666 | 114.64 |

**Zero-shot: macro-F1 trên val theo ngưỡng T** (`data/results/r05/grid_clip_clip_zs.csv`; xác suất softmax cao nhất < T → `unknown`):

| T | 0 (không gán unknown) | 0.5 | **0.6** | 0.7 | 0.8 | 0.9 |
|---|---|---|---|---|---|---|
| Val macro-F1 | 56.2 | 60.3 | **60.5** | 60.3 | 59.9 | 59.5 |

Hai giá trị tham khảo T = 0.7 (60.3) và T = 0.8 (59.9) gần như ngang T = 0.6 (60.5): đường cong rất phẳng từ 0.5 đến 0.9; chỉ việc **có** ngưỡng (so với T = 0) là tăng rõ (+4.3).

Nhận xét:
- **Embedding CLIP + bộ phân loại có giám sát vượt xa đặc trưng thủ công**: CLIP + SVM RBF test 88.9 so với 46.4 của SIFT-BoVW + SVM (gần gấp đôi), và cả 4 mô hình trên CLIP đều ≥ 83.3. **Linear probe** đúng nghĩa (bộ phân loại tuyến tính = CLIP + LogReg) đạt 87.6, đã hơn mọi cấu hình thủ công; SVM RBF, RF, XGBoost là bộ phân loại phi tuyến, không phải linear probe. Đây là mức trần tham khảo, kèm rủi ro dữ liệu ở trên.
- **Zero-shot kém hơn nhiều so với có giám sát** (test 58.1 so với 88.9) nhưng vẫn hơn mọi cấu hình thủ công (46.4) mà **không dùng nhãn nào để học** (chỉ chọn T trên val).
- **Mô hình trên CLIP xếp hạng như trên SIFT-BoVW**: SVM > XGBoost ≈ LogReg > RF; SVM thắng với C = 10 (biên lưới).
- **Val khớp test** ở mọi cấu hình CLIP (chênh ≤ 2.4).

#### Đánh giá từng lớp (test, CLIP)

P = precision, R = recall. Đầy đủ trong notebook mục 8.

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

## Thời gian chạy

Máy: Apple Silicon (arm64), 10 CPU (4 nhân hiệu năng cao + 6 nhân tiết kiệm điện), RAM 24 GB, `n_jobs=-1`. Chỉ ghi phiên bản mới nhất; thời gian `r04`, `f02` / `r03`, `r02` và `f01` / `r01` xem [`EXPERIMENTS.md`](EXPERIMENTS.md).

### `f03` (bước 1, 12,833 ảnh, 24/10/2026)

**Bước 1** (`f03`, 10 khối thủ công + CLIP, `--clip`): tổng thời gian thực **400.6 s (6.7 phút)**; RAM đỉnh **4,008 MB** (tăng thêm 3,833 MB), CPU 2,660 s (6.6 nhân trung bình); đĩa 585.6 MB.

| Công đoạn | Thời gian thực (s) |
|---|---|
| Lấy mẫu descriptor + MiniBatchKMeans K=500 (chỉ train) | 77.1 |
| Dense SIFT + mã hoá BoVW (cả 3 split) | 137.3 |
| Đặc trưng ảnh hog/lbp/color/hog_upper/gist/skin (cả 3 split) | 29.3 |
| geom | 0.2 |
| **CLIP ViT-B/32** (cả 3 split, CPU, 8 luồng torch) | **151.7** (train 106.9, val 22.0, test 22.8) |

- CLIP: ~85 ảnh/giây (9,069 ảnh train trong 107 s), RAM đỉnh công đoạn 3.9 GB, 2.2 nhân trung bình. Khối CLIP 26.3 MB trên đĩa (512 chiều float32). Lần đầu tải trọng số ~350 MB từ Hugging Face (thời gian tải nằm trong 151.7 s, không tách riêng).
- So với ước tính trước khi chạy (thủ công ~4 phút + CLIP 3.5–5 phút): thực tế 6.7 phút, CLIP nhanh hơn ước tính (2.5 phút). Ước tính CLIP dựa trên lần chạy thử 60 ảnh, gồm cả thời gian nạp mô hình nên tốc độ bị đánh giá thấp.
- Khối thủ công của `f03` **giống từng bit** `f02` (đã so SIFT-BoVW, HOG trên test): pipeline tái lập được.

### `r05` (bước 2–4 trên `f03`, 31/10/2026)

Lệnh: `FEATS="hog lbp color sift clip" MODELS="logreg svm rf xgb clip_zs" bash run_all.sh f03 r05 "..."` (dùng lại `f03`, không trích lại). Cả pipeline **31 phút 56 s** (22:34:38 → 23:06:34), so với ước tính ~35 phút.

| Bước | Thời gian thực | RAM đỉnh (MB) | RAM tăng thêm (MB) | CPU (s) | Số nhân dùng trung bình |
|---|---|---|---|---|---|
| 2. Chọn mô hình (21 cấu hình) | **1,622.6 s (27.0 phút)** | 6,743 | 6,593 | 9,082 | 5.6 |
| 3. Test (21 cấu hình, fit trên train+val) | 220.0 s | 1,723 | 1,548 | 958 | 4.4 |
| 4. Notebook | ~70 s (ước tính = tổng − bước 2–3) | – | – | – | – |

- **Lưu ý đo đạc**: trong lúc bước 2 chạy, notebook báo cáo được chạy song song 3 lần (22:39–22:41, 22:45–22:48, 23:00–23:02) để sửa báo cáo, nên thời gian và RAM của các cấu hình chạy trong các khoảng đó có thể cao hơn khi chạy riêng.
- **Bước 2 theo cấu hình** (giây): Color-HSV + SVM 259.8, HOG + RF 250.6, Color-HSV + XGBoost 196.5, CLIP + XGBoost 154.3, SIFT-BoVW + SVM 109.4, CLIP + RF 109.2, HOG + XGBoost 96.6, Color-HSV + RF 77.7, SIFT-BoVW + XGBoost 60.7, CLIP + SVM 60.0, LBP + SVM/XGBoost/RF ~48, HOG + SVM 48.1, SIFT-BoVW + RF 29.9, HOG + LogReg 15.3; LogReg còn lại < 5; CLIP zero-shot 0.2.
- RAM đỉnh bước 2: 6.7 GB (HOG + SVM), thấp hơn `r03` (7.8 GB, HOG + RF).
- Dung lượng `data/results/r05/` 701 MB, mô hình 733 MB theo `model_mb` (gồm cả RF, mỗi mô hình RF 69–115 MB).

## Thực nghiệm

Chỉ ghi **thực nghiệm mới nhất**. Toàn bộ nhật ký (mục 1–8), kết luận tạm và [bài học kinh nghiệm](EXPERIMENTS.md#bài-học-kinh-nghiệm) nằm trong [`EXPERIMENTS.md`](EXPERIMENTS.md).

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

## Cần làm

**Trạng thái hiện tại (để tiếp tục lần sau):**
- `.venv` đã cài đủ thư viện, `libomp` đã cài.
- `data/features/f02/` có đủ 10 khối đặc trưng cho bộ gộp (12,833 ảnh, `labels_split_unk3000.csv`). **Không cần chạy lại bước 1** khi chỉ đổi mô hình hoặc lưới: chạy `02_train_select.py --feat_version f02 --version r03 ...`.
- `data/features/f03/` = `f02` + khối CLIP. `data/results/r05/` là **kết quả mới nhất** (4 đặc trưng thủ công × 4 mô hình + CLIP × 5, trùng chính xác `r03` + `r04`); notebook hiển thị `r05`.
- `data/results/r04.zip` (96 MB, 27/10/2026, có thư mục rác `__MACOSX/`) không do script sinh ra và không phải một phiên bản; nên chuyển ra ngoài `data/results/` (chưa làm, chờ người dùng quyết định).
- `torch` + `open_clip_torch` đã cài trong `.venv` (chỉ thêm gói mới, numpy/sklearn/xgboost giữ nguyên phiên bản).
- `data/features/f01/`, `data/results/r01/` là bản cũ trên 519 ảnh V-COCO, giữ để tham khảo.

Việc bắt buộc, làm trước:
- [ ] **Lấy đủ dữ liệu**: hiện chỉ có 519/15,187 crop trong `data/vcoco_6class_crops/`. Kiểm tra bằng `find data/vcoco_6class_crops/train -name '*.jpg' | wc -l`, phải ra 6,803.
- [ ] **Quyết định có chuyển sang `data/dataset_6class_crops/` (22,366 ảnh, 4 nguồn) hay không** (xem mục Dữ liệu). Bộ này có số ảnh hành động nhiều hơn hẳn bộ V-COCO, nhưng chưa có split, nên muốn dùng thì phải:
  - [x] tạo cột `split`: đã có `labels_split.csv` (70/15/15 theo ảnh gốc, seed 2026; xem mục Dữ liệu). Cách này không giữ test gốc của v_coco;
  - kiểm tra xem có ảnh nào ở nguồn khác trùng với ảnh test của v_coco không, tránh lộ test. README dữ liệu chỉ ghi đã bỏ các cặp trùng *cùng một người*;
  - [x] trỏ pipeline sang dữ liệu mới và tạo phiên bản mới: đã chạy `f02`/`r02` (thực nghiệm 5).
- [x] Chạy bước 1 → 4 thành phiên bản `f01` / `r01` (`run_all.sh`).
- [x] Thêm phần "Thời gian chạy" vào `04_report.ipynb` (mục 5).
- [x] Viết lại phần nhận xét (mục 7) trong notebook theo kết quả thật: đã viết cho `r03` (31/10/2026); bỏ phần giới thiệu và minh hoạ các đặc trưng bổ sung chưa dùng.
- [x] **Chạy `r05` gồm cả 4 đặc trưng thủ công và CLIP trên `f03`** (xong 31/10/2026): 31 phút 56 s; số trùng chính xác `r03` + `r04`; notebook hiển thị `r05`, CLIP tách riêng ở mục 8.

Cải thiện tiếp từ `r03`:
- [x] **Chạy CLIP (`f03`, `r04`)** (xong, thực nghiệm 8): CLIP + SVM RBF test 88.9, linear probe CLIP + LogReg 87.6, zero-shot 58.1.
- [ ] **Zero-shot: thêm prompt cho `unknown`** (ví dụ "a photo of a person standing", "a photo of a person walking") và **prompt ensembling** (nhiều câu mỗi lớp, lấy trung bình embedding). Lý do: 65% ảnh `unknown` bị đoán `ride` vì không có prompt `unknown`; ngưỡng T không sửa được. Rẻ: chỉ cần tính lại `clip_text.npy`, không cần trích lại ảnh.
- [ ] **Ghép CLIP với đặc trưng thủ công** (nối CLIP + HOG, hoặc late fusion CLIP + SVM với HOG + SVM): đặc trưng thủ công có bổ sung gì cho CLIP không? Đây là câu hỏi duy nhất còn ý nghĩa cho phần thủ công khi đặt cạnh CLIP.
- [ ] **Kiểm tra rủi ro CLIP đã thấy ảnh test**: so kết quả CLIP + SVM theo nguồn ảnh test (v_coco/COCO so với stanford40, pascal_voc, hico_det). Nếu v_coco cao bất thường so với nguồn khác thì có dấu hiệu lộ dữ liệu.
- [x] **Đo RAM**: bước 2–3 đã đo ở `r03` (đỉnh 7.8 GB ở HOG + RF). Còn thiếu RAM bước 1: sẽ có khi trích `f03`.
- [ ] **Tách nguồn RAM của bước 2** (nền worker, bản sao ma trận, cây RF, cache kernel SVM) trước khi chạy bộ phối lớn (`fused_all` ~9,300 chiều): RAM đỉnh `r03` 7.8 GB với HOG 3,780 chiều.
- [x] **Chạy `r03` để có đủ chỉ số trong csv** (xong, thực nghiệm 7): code đã ghi accuracy, precision, F1 từng lớp (val và test) và confusion matrix số đếm (`confusion_*.csv`), nhưng chưa chạy. Có thể gộp vào lần chạy `r03` kế tiếp (ví dụ mở rộng lưới C bên dưới). Notebook đã sửa (tính đủ chỉ số và vẽ confusion số đếm cho mọi cấu hình đã test, chạy được cả với `r01`/`r02`) nhưng output đang lưu vẫn là bản cũ; Run All khi có `r03`.
- [ ] **Mở rộng lưới ở biên**: LogReg C về phía nhỏ (0.001, 0.003; C = 0.01 nhỏ nhất thắng ở HOG, Color-HSV, SIFT-BoVW), SVM C về phía lớn (30, 100; C = 10 thắng ở HOG), XGBoost thêm 1,000 cây (600 thắng ở 3/4 đặc trưng). LogReg rẻ (~20 s); SVM, XGBoost thêm vài phút mỗi đặc trưng (ước tính).
- [ ] **Ghép HOG + SIFT-BoVW cho SVM** (nối đặc trưng, PCA 256 vì > 1,500 chiều) hoặc **late fusion HOG + SVM và SIFT-BoVW + SVM**: ở `r03`, HOG + SVM mạnh ở `hold`, `ride` (dáng người), SIFT-BoVW + SVM mạnh ở `talk_on_phone`, `work_on_computer` (vật). Chọn trọng số trên val.
- [ ] Cải thiện lớp `hold` (recall test 23.7–28.7 với LogReg ở `r02`; 27.4 với SIFT-BoVW + SVM, 46.6 với HOG + SVM ở `r03`). Đã xem confusion (mục Đánh giá từng lớp): `hold` bị nhầm đều sang `eat/drink`, `talk_on_phone`, `work_on_computer`, và 48% lỗi nằm trong 4 lớp "tay cầm vật". Hướng thử: đặc trưng giữ vị trí vật so với đầu/tay (SIFT spatial pyramid, `upper` thân trên) thay cho BoVW toàn ảnh. Lý do: lỗi đến từ việc không phân biệt được vật ở đâu, không phải do thiếu dữ liệu.
- [ ] Xem bằng mắt các ảnh `eat/drink` → `work_on_computer` và `unknown` → `ride` để kiểm chứng giả thuyết "do bối cảnh" (bàn ăn/bàn làm việc, người đứng cạnh xe).
- [ ] ~~Late fusion SIFT-BoVW + LogReg và SIFT-BoVW + RF~~: thay bằng late fusion HOG + SVM và SIFT-BoVW + SVM ở trên (RF kém hơn SVM ở `r03`).

Cải thiện mô hình và đặc trưng (chưa làm):
- [ ] **Fisher Vector / VLAD** thay cho BoVW để mã hoá SIFT mạnh hơn.
- [ ] **Embedding CNN pretrain ImageNet** (ResNet50, EfficientNet; đã có code mẫu trong `code/extract_image_and_embedding.ipynb`) đưa qua 4 mô hình ML, làm mốc so sánh. Không dùng backbone pretrain trên COCO, vì nó đã thấy ảnh test.
- [ ] Thử tham số đặc trưng: `BOVW_K` (200/1000), spatial pyramid thêm mức 4×4, `UPPER_FRAC` (0.3/0.5), mức nới SIFT, chuẩn hoá từng khối trước khi nối. Lý do: các giá trị hiện tại được chọn theo kinh nghiệm, chưa thử; riêng chuẩn hoá từng khối vì khi nối, khối nhiều chiều (HOG 3,780-d) lấn át khối ít chiều (`geom` 8-d), trong khi `geom` lại là đặc trưng mạnh nhất.
- [ ] **Late fusion**: lấy trung bình xác suất của các mô hình tốt nhất trên từng đặc trưng (cần `SVC(probability=True)`), so sánh với nối đặc trưng (early fusion). Lý do: SVM/RF/XGBoost trên bộ nối lớn hay sụp về đoán toàn `unknown`, trong khi từng đặc trưng riêng lẻ vẫn học được; gộp ở mức xác suất tránh được vấn đề số chiều.
- [ ] Chỉnh ngưỡng quyết định cho lớp `unknown` trên val để tăng recall các lớp hành động. Lý do: `unknown` chiếm ~65%, và trên test của `r01`, recall `unknown` là 56–100 trong khi recall các lớp hành động chỉ 0–50.
- [ ] Nếu val vẫn nhiễu: chọn siêu tham số bằng `StratifiedGroupKFold` trên train+val (nhóm theo `image_id`) thay cho 1 tập val cố định.
- [ ] Cờ lọc crop nhỏ `MIN_SIDE` đã có trong `config.py` nhưng **chưa được dùng**: cần nối vào `01_extract_features.py` (chỉ lọc ở train) nếu muốn thử. Lý do: 16 crop có cạnh ngắn < 32 px (README dữ liệu), khi phóng lên 64×128 / 128×256 thì HOG, LBP gần như chỉ còn nhiễu.
- [ ] Nếu bước 2 quá chậm với đủ dữ liệu: thu hẹp lưới SVM/XGBoost, hoặc chạy theo nhóm bằng `--feats ... --models ...`. Lý do: SVM RBF tăng khoảng O(n²) theo số ảnh, train đủ lớn gấp ~26 lần mẫu hiện tại (ước tính hàng chục phút đến vài giờ).

## Tài liệu tham khảo

- Li et al., *Disentangled Pre-training for Human-Object Interaction Detection* (DP-HOI), CVPR 2024: https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Disentangled_Pre-training_for_Human-Object_Interaction_Detection_CVPR_2024_paper.pdf

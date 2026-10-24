# CLAUDE.md: Đồ án nhận diện hành động người–vật (V-COCO, 6 lớp)

Trả lời và ghi chú bằng **tiếng Việt**. Đọc `README.md` trước khi làm: cách chạy, kết quả hiện tại, thực nghiệm mới nhất và **Cần làm**; nhật ký thực nghiệm đầy đủ và bài học kinh nghiệm ở `EXPERIMENTS.md`.

## Quy tắc bắt buộc khi thực nghiệm

File **`EXPERIMENTS.md`** ghi (đầy đủ, mọi mục đánh số) **kết quả chạy các đặc trưng, mô hình khác nhau nhằm cải thiện kết quả mô hình**, để người làm đồ án hiểu **vì sao** phải thay đổi. Áp dụng cho: đặc trưng, tham số đặc trưng, mô hình, lưới siêu tham số, cách phối đặc trưng, tiền xử lý, cách đánh giá / chọn mô hình, và thay đổi hiệu năng có ảnh hưởng tới cấu hình mô hình.

**Không** ghi vào Thực nghiệm những việc không liên quan tới cải thiện mô hình: thêm/xoá/di chuyển file, cách lưu phiên bản, quy ước notebook, sửa tài liệu, sửa lỗi công cụ. Những việc đó (nếu cần) chỉ cập nhật ở mục Cách chạy / Phiên bản kết quả / Cài đặt.

1. **Đề xuất trước, giải thích lý do.** Khi đề xuất một thay đổi, nêu rõ:
   - vấn đề quan sát được (kèm số đo: macro-F1, recall từng lớp, thời gian chạy…) hoặc giả thuyết dựa trên đặc điểm dữ liệu;
   - thay đổi cụ thể là gì (file, tham số, giá trị cũ → mới);
   - kỳ vọng sẽ cải thiện điều gì, và cái giá phải trả (thời gian, số chiều, rủi ro).
2. **Sau khi làm, ghi vào `EXPERIMENTS.md`** thành một mục đánh số mới (nối tiếp số cũ), theo mẫu dưới; đồng thời **thay** mục ở `## Thực nghiệm` trong `README.md` bằng mục mới này (README chỉ giữ thực nghiệm mới nhất):
   ```
   N. **<Tên thay đổi>** (<ngày>)
      - Vấn đề / lý do: ...
      - Thay đổi: ... (file, tham số cũ → mới)
      - Kết quả: trước → sau (val macro-F1, test nếu có, thời gian chạy)
      - Nhận xét / tác dụng phụ: ...
   ```
   Ghi đúng sự thật: nếu cải thiện đến từ nguyên nhân khác dự kiến, hoặc không cải thiện, cũng phải ghi. Không ghi số chưa đo; số ước tính phải ghi rõ "ước tính".
3. **Kinh nghiệm rút ra về đặc trưng / mô hình** (cạm bẫy khi train, mẹo tăng tốc mô hình, cách chọn mô hình) → thêm vào `## Bài học kinh nghiệm` trong `EXPERIMENTS.md`. Ghi chú môi trường / cài đặt thì đưa vào mục Cài đặt.
4. **Việc đề xuất mà chưa làm** → thêm vào `## Cần làm` kèm lý do, để lần sau làm tiếp. Việc đã xong thì đánh dấu `[x]` hoặc chuyển sang Thực nghiệm.
5. **Thời gian chạy**: mỗi lần chạy, cập nhật mục `## Thời gian chạy` (các file `data/features/<fXX>/timing_extract.csv`, cột thời gian trong `val_results.csv` / `test_results.csv`, `total_wall_s` trong `run_info.json`).
6. Cập nhật bảng kết quả trong README và chạy lại notebook `code/04_report.ipynb` khi kết quả thay đổi; ghi rõ chạy trên bao nhiêu ảnh.
7. **Phiên bản**: mỗi lần chạy tạo phiên bản mới (`data/features/fXX`, `data/results/rXX`, kèm `--note`). Thêm ngay 1 dòng vào bảng `## Phiên bản kết quả` trong README (phiên bản, ngày, số ảnh, nội dung, kết quả chính, số thực nghiệm nếu có, ghi chú), và nhắc số phiên bản trong mục thực nghiệm tương ứng (`EXPERIMENTS.md`).

## Quy tắc về dữ liệu và đánh giá

- Chọn mô hình và siêu tham số **chỉ trên val**; **test chỉ để báo cáo**, không dùng để chọn.
- Đánh giá bằng **macro-F1 + recall từng lớp** (lớp `unknown` chiếm ~65%); không dùng accuracy làm chỉ số chính.
- Cái gì học từ dữ liệu (KMeans codebook, scaler, PCA) **chỉ fit trên train** (hoặc train+val ở bước test cuối).
- Không dùng backbone/detector pretrain trên COCO (đã thấy ảnh test); ImageNet thì được.
- Không dùng cột `original_label`, `hold_object` trong `labels.csv` làm đặc trưng (lộ nhãn).

## Quy tắc thao tác

- Chạy bằng `.venv/bin/python` trong thư mục `code/`, hoặc cả pipeline bằng `bash run_all.sh <fXX> <rXX> "ghi chú"` ở thư mục gốc. Trên macOS, XGBoost cần `brew install libomp`.
- **Không bao giờ xoá hay ghi đè kết quả đã chạy** (`data/results/`, `data/features/`, dữ liệu, notebook đã chạy), kể cả kết quả dở dang hay sinh ra do nhầm: chuyển vào một phiên bản có ghi chú thay vì xoá. Chỉ xoá khi người dùng yêu cầu rõ ràng.
- Không dùng `--update` (bổ sung vào phiên bản cũ) nếu chưa được đồng ý; mặc định tạo phiên bản mới. Chạy lại một phần: `02_train_select.py --feat_version fXX --version rXX --update --feats ... --models ...`.
- Notebook `code/04_report.ipynb` là **nơi thể hiện kết quả chính**, luôn hiển thị **phiên bản mới nhất**. Sau mỗi phiên bản mới: sửa giá trị mặc định `VERSION` ở ô đầu thành `rXX`, rồi Run All (`jupyter nbconvert --to notebook --execute --inplace 04_report.ipynb`).
- `data/results/<rXX>/` **chỉ chứa kết quả do script sinh ra** (csv, txt, png, npy, joblib, `run_info.json`). Không tạo bản sao notebook (`.ipynb`) hay `.html` trong `data/results/`. Kết quả của các phiên bản cũ xem lại bằng cách đặt `VERSION` cũ rồi chạy notebook. Ngày 12/10/2026, theo yêu cầu người dùng, `features/` và `results/` cũ (f01, f01_repro_check, r04_xgb_pca, r05_interrupted, r06) đã được chuyển ra `../_backup_DoAn_20261012/` và chạy lại từ đầu thành `f01`/`r01` bằng `run_all.sh`.
- Hỏi trước khi chạy lại các bước tốn thời gian (bước 2 với đủ dữ liệu có thể mất hàng chục phút trở lên).
- Thêm bộ đặc trưng mới: thêm khối vào `features.py` / `config.ATOMIC_FEATURES`, rồi thêm dòng vào `FEATURE_SETS` trong `code/models.py`.

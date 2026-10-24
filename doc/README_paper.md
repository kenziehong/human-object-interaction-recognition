# Tóm tắt bài báo DP-HOI: input, output, metric

Li et al., *Disentangled Pre-training for Human-Object Interaction Detection* (DP-HOI), CVPR 2024.
File: `Li_Disentangled_Pre-training_for_Human-Object_Interaction_Detection_CVPR_2024_paper.pdf` (11 trang, chỉ có bài chính, **không có supplementary material**). File PDF không đưa lên repo; tải bản Open Access tại https://openaccess.thecvf.com/content/CVPR2024/papers/Li_Disentangled_Pre-training_for_Human-Object_Interaction_Detection_CVPR_2024_paper.pdf rồi đặt vào `doc/`.
Code: https://github.com/xingaoli/DP-HOI

> Ghi chú: phần định nghĩa chi tiết của metric (mục 3) là **cách đo chuẩn của HICO-DET [6] và V-COCO [7]**; bài báo chỉ nêu tên metric, không định nghĩa lại. Những chỗ bài báo không ghi được đánh dấu "bài báo không ghi".

## 1. Bài toán

Bài báo làm **phát hiện HOI** (HOI detection): với mỗi cặp người–vật có tương tác trong ảnh, phải tìm bbox người, bbox vật, lớp vật và động từ.

Bài báo tách bài toán này thành **2 bài toán con** (mục 1, 3.1: *"interactive human-object pair detection and interaction classification"*), ứng với 2 nhánh khi tiền huấn luyện (pre-training):

| Bài toán con | Nhánh trong DP-HOI | Dữ liệu tiền huấn luyện |
|---|---|---|
| 1. Phát hiện người và vật | Object detection branch (detection decoder) | COCO (117.266 ảnh, 80 lớp), Objects365 (117.266 ảnh, 365 lớp), có bbox |
| 2. Phân loại tương tác (động từ) | Verb classification branch (detection decoder dùng chung tham số + interaction decoder) | Haa500 (52.644 frame, 500 lớp), Kinetics-700 (117.266 frame, 700 lớp), Flickr30k (25.977 ảnh), VG (54.280 ảnh); chỉ có nhãn mức ảnh/video hoặc caption |

Sau khi tiền huấn luyện, trọng số được dùng để khởi tạo các mô hình phát hiện HOI có sẵn (CDN-S, QPIC, GEN, HOICLIP, UPT, PViC), rồi fine-tune trên HICO-DET hoặc V-COCO.

**Lưu ý:** khi tiền huấn luyện, nhánh 2 **không** phân loại tương tác của một cặp người–vật. Các tập hành động không có nhãn cặp người–vật, nên tác giả rút gọn thành *nhận diện mọi hành động của một người* (mục 3.3). Chỉ ở bước fine-tune, mô hình đích mới phân loại động từ cho từng cặp.

## 2. Input và output

### 2.1. Nhánh phát hiện (bài toán con 1)

- **Input:** cả bức ảnh. Ảnh được resize sao cho cạnh ngắn 800 px, cạnh dài tối đa 1.333 px. Mạng gồm ResNet-50 → transformer encoder → detection decoder với N = 100 object query.
- **Output:** với mỗi query có bbox (ŷ_box) và lớp vật (ŷ_o). Dự đoán được ghép với nhãn bằng bipartite matching như DETR.
- **Loss:** L_d = 5·L1 + 2·GIoU + 1·cross-entropy.

### 2.2. Nhánh phân loại động từ (bài toán con 2)

- **Input:** cả bức ảnh (hoặc frame video), **không phải ảnh crop người**. Frame Kinetics-700 được resize về 256×256; các tập khác resize như nhánh phát hiện. Nhãn chỉ có **ở mức ảnh** (danh sách động từ trong ảnh), không có bbox người và không biết người nào làm hành động nào.
- **Xử lý trong mạng:**
  1. Detection decoder tìm người. Query nào có điểm "person" > T = 0,9 thì trở thành **Reliable Person Query (RPQ)**.
  2. Mỗi RPQ đi qua interaction decoder và tìm dấu hiệu hành động của đúng người đó bằng cross-attention.
- **Output:**
  - ŷ_a ∈ R^(Np × Ca): điểm động từ cho từng người, với Np là số RPQ và Ca là tổng số động từ của mọi tập hành động.
  - Đây là bài toán **đa nhãn** (multi-label): mỗi động từ là một bộ phân loại nhị phân. Mạng chỉ bật các động từ có trong tập dữ liệu mà ảnh đó thuộc về, vì các tập có thể trùng nghĩa động từ với nhau.
  - **Verb-wise Prediction Fusion (VPF):** lấy max-pooling theo từng động từ qua mọi người trong ảnh (với video thì qua mọi frame), được một vector dự đoán cho cả ảnh, rồi áp nhãn mức ảnh lên vector này.
  - Với dữ liệu ảnh–caption: không dùng xác suất từng lớp mà dùng embedding của RPQ, so khớp với embedding văn bản CLIP của câu "a photo of {human} {verb} {object}". Bộ ba HOI được lấy từ caption bằng bộ phân tích ngữ pháp theo luật. Mẫu âm: gom thành 100 cụm, lấy 10 loại HOI mỗi cụm.
- **Loss:** L_v = 1·focal + 1·InfoNCE, trong đó InfoNCE tính hai chiều: L_s = ½(L_i2t + L_t2i). Loss tổng: L = L_d + λ_v·L_v, với λ_v = 1.

### 2.3. Bài toán đích sau fine-tune (phát hiện HOI)

- **Input:** cả bức ảnh từ HICO-DET hoặc V-COCO.
- **Output:** danh sách bộ ⟨bbox người, bbox vật, lớp vật, động từ, điểm tin cậy⟩.

| Bộ dữ liệu | Số ảnh train / test | Số lớp vật | Số động từ | Số loại HOI |
|---|---|---|---|---|
| HICO-DET | 38.118 / 9.658 | 80 | 117 | 600 (138 rare) |
| V-COCO | 5.400 / 4.946 | 80 | 26 | bài báo không ghi |

## 3. Metric

Bài báo **chỉ dùng mAP của bài toán phát hiện HOI**. Mọi bảng số (Bảng 2–8) và Hình 1 đều là mAP trên HICO-DET hoặc AP_role trên V-COCO.

**Không có metric nào cho riêng từng bài toán con.** Bài báo không báo cáo mAP phát hiện vật, cũng không báo cáo top-1, accuracy hay mAP phân loại động từ, kể cả trên Haa500 hay Kinetics-700. Các ablation về chính nhánh phân loại động từ (Bảng 6: bỏ VPF/RPQ; Bảng 7: max- so với avg-pooling; Bảng 8: cách khởi tạo) đều đo gián tiếp: tiền huấn luyện, fine-tune CDN-S trên HICO-DET, rồi báo cáo mAP phát hiện.

### 3.1. mAP trên HICO-DET

**Khi nào một dự đoán được tính là đúng (true positive):** dự đoán phải thoả cả ba điều kiện:
- đúng loại HOI (động từ + lớp vật, ví dụ *ride bicycle*);
- bbox người **và** bbox vật đều có IoU ≥ 0,5 với một cặp trong nhãn;
- cặp nhãn đó chưa được dự đoán khác nào khớp trước (nếu trùng thì dự đoán sau tính là false positive).

**Cách tính:** với mỗi loại HOI, sắp các dự đoán theo điểm tin cậy, vẽ đường precision–recall rồi tính AP. **mAP** là trung bình đều AP của các loại HOI.

| Thiết lập | Nghĩa | Bảng |
|---|---|---|
| **DT mode** (Default) | Tính AP trên toàn bộ ảnh test. HICO-DET còn có chế độ *Known Object*, nhưng bài báo không dùng | 2, 5–8 |
| **Full** | Trung bình trên cả 600 loại HOI | 2, 5–8 |
| **Rare** | 138 loại HOI có < 10 mẫu train | 2, 5–8 |
| **Non-Rare** | 462 loại còn lại | 2, 5–8 |
| **UV** (Unseen Verb) | Bỏ một số động từ khỏi tập train rồi kiểm tra trên các động từ đó | 3 |
| **RF-UC** (Rare First, Unseen Composition) | Bỏ một số tổ hợp động từ–vật khỏi train, ưu tiên bỏ tổ hợp hiếm | 3 |
| **NF-UC** (Non-rare First) | Như trên nhưng ưu tiên bỏ tổ hợp phổ biến | 3 |

- Ở các thiết lập zero-shot, mAP được báo cáo trên 3 nhóm: **Unseen** (loại bị bỏ khi train), **Seen** (loại có trong train) và **Full** (tất cả).
- Bài báo không ghi số động từ hay số tổ hợp bị bỏ. Theo chuẩn thường dùng (VCL, GEN-VLKT), mỗi thiết lập UC bỏ 120 tổ hợp, nhưng con số này **chưa kiểm chứng** trong bài báo.

### 3.2. AP_role, Scenario 1 trên V-COCO

Bài báo chỉ ghi: *"We use the mean average precision of Scenario 1 (AP_role) [7]"*.

- **AP_role (role AP):** một dự đoán đúng khi đúng hành động, bbox người có IoU ≥ 0,5, **và** bbox của vật ở vai trò đó cũng có IoU ≥ 0,5. Ví dụ với *eat*, vai trò `obj` là thức ăn, vai trò `instr` là dụng cụ. AP được tính cho từng cặp hành động–vai trò rồi lấy trung bình.
- **Scenario 1 so với Scenario 2:** khác nhau ở trường hợp nhãn không có vật cho vai trò đó (vật bị che hoặc nằm ngoài ảnh).
  - Scenario 1 (bài báo dùng, chặt hơn): mô hình phải dự đoán bbox vật rỗng thì mới tính là đúng.
  - Scenario 2: bỏ qua bbox vật trong trường hợp này.
- Bài báo không ghi AP_role được trung bình trên bao nhiêu cặp hành động–vai trò.

### 3.3. Cách bài báo báo cáo kết quả

- Mọi cải thiện đều tính bằng **điểm mAP tuyệt đối**. Ví dụ "+3,02% mAP" nghĩa là CDN-S† tăng từ 31,98 lên 35,00, không phải tăng 3,02% tương đối.
- Hình 1 là đường **mAP theo số epoch**, dùng để so tốc độ hội tụ.
- Hình 3 chỉ là ảnh minh hoạ attention map, không có con số.

### 3.4. Kết quả chính

| Bộ dữ liệu | Mô hình | Không có DP-HOI | Có DP-HOI |
|---|---|---|---|
| HICO-DET (DT, Full / Rare / Non-Rare) | CDN-S† | 31,98 / 28,61 / 32,99 | 35,00 / 32,38 / 35,78 |
| HICO-DET (DT, Full / Rare / Non-Rare) | HOICLIP | 34,69 / 31,12 / 35,74 | 36,56 / 34,36 / 37,22 |
| V-COCO (AP_role, S1) | CDN-S | 61,7 | 64,8 |
| V-COCO (AP_role, S1) | GENs | 62,4 | 66,6 |

## 4. So với đồ án

| | DP-HOI | Đồ án |
|---|---|---|
| Bài toán | Phát hiện HOI (định vị + phân loại) | Chỉ phân loại hành động, gần với bài toán con 2 ở dạng rút gọn khi tiền huấn luyện |
| Input | Cả ảnh, mạng tự tìm người | Ảnh crop 1 người, đã có sẵn bbox |
| Output | Đa nhãn: bbox người + bbox vật + 26/117 động từ | Đơn nhãn, 6 lớp, có lớp `unknown`; không dự đoán bbox vật |
| Mô hình | DETR, ResNet-50, trọng số pretrain trên COCO | Đặc trưng thủ công + ML cổ điển |
| Metric | mAP / AP_role | Macro-F1 + recall từng lớp |
| Đơn vị tính metric | AP cho từng loại HOI / từng vai trò | F1 và recall cho từng lớp |
| Gộp metric | Trung bình đều các loại (mAP) | Trung bình đều các lớp (macro) |
| Cần định vị đúng | Có (IoU ≥ 0,5 cho cả người và vật) | Không |
| Dùng điểm tin cậy | Có (xếp hạng để vẽ đường PR) | Không (chỉ dùng nhãn dự đoán cuối) |

- **Không so số trực tiếp được** với bài báo: khác bài toán, khác số lớp, khác metric, và bài báo không có số phân loại riêng.
- Cả hai cùng **lấy trung bình đều giữa các lớp**, nên lớp hiếm có trọng số ngang lớp phổ biến. Điều này cùng tinh thần với việc đồ án không dùng accuracy.
- Bài báo dùng trọng số DETR pretrain trên COCO. Đồ án không được dùng loại trọng số này, vì ảnh test V-COCO lấy từ COCO.
- Nếu muốn có một chỉ số gần với bài báo hơn, đồ án có thể báo cáo thêm AP từng lớp và mAP theo kiểu one-vs-rest từ xác suất dự đoán (`sklearn.metrics.average_precision_score`). Việc này **chưa làm**.

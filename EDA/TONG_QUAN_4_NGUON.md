# Tổng quan 4 nguồn dữ liệu (dữ liệu gốc, trước khi map sang 6 lớp)

Số liệu lấy từ output của 4 notebook trong thư mục `EDA/`. Riêng các ô ghi *(bản gốc)* lấy theo trang chủ hoặc bài báo của bộ dữ liệu, vì notebook không chạy trên toàn bộ dữ liệu.

## Bảng tổng hợp

| | **Stanford 40 Actions** | **PASCAL VOC 2012 Action** | **HICO-DET** | **V-COCO** |
|---|---|---|---|---|
| **1. Tổng số ảnh** | **3.389** ảnh (bản Meta-Album trên OpenML mà notebook dùng)<br>*(bản gốc: 9.532 ảnh)* | **4.588** ảnh, chứa **6.278** người có nhãn (split `trainval`) | **3.000** ảnh (notebook chỉ lấy 3.000 ảnh đầu của split train để EDA)<br>*(bản gốc: 47.776 ảnh = 38.118 train + 9.658 test)* | **10.346** ảnh, chứa **16.199** người<br>(trainval 5.400 ảnh / 8.431 người; test 4.946 ảnh / 7.768 người) |
| **2. Gồm những nhãn nào** | **40** hành động, ví dụ `riding_a_bike`, `phoning`, `drinking`, `using_a_computer`, `holding_an_umbrella`… (danh sách đầy đủ bên dưới) | **11** hành động: `jumping`, `phoning`, `playinginstrument`, `reading`, `ridingbike`, `ridinghorse`, `running`, `takingphoto`, `usingcomputer`, `walking`, `other` | **600** cặp tương tác (vật, động từ), ghép từ **80** loại vật COCO × **117** động từ. Ví dụ `(cell_phone, talk_on)`, `(horse, ride)`, `(cup, drink_with)` | **26** hành động: `hold`, `stand`, `sit`, `ride`, `walk`, `look`, `hit`, `eat`, `jump`, `lay`, `talk_on_phone`, `carry`, `throw`, `catch`, `cut`, `run`, `work_on_computer`, `ski`, `surf`, `skateboard`, `smile`, `drink`, `kick`, `point`, `read`, `snowboard` |
| **3. Đơn nhãn hay đa nhãn** | **Đơn nhãn**: mỗi **ảnh** có đúng 1 nhãn | **Đa nhãn theo người**, nhưng hầu hết là 1 nhãn: chỉ **68 / 6.278** người (1,1%) có ≥ 2 hành động | **Đa nhãn**: trung bình **1,82** cặp HOI / ảnh; 1 người thường có nhiều cặp (ví dụ vừa `ride` vừa `straddle` xe máy) | **Đa nhãn**: **14.346 / 16.199** người (≈ 89%) có ≥ 2 hành động, thường là 1 tư thế (`stand`, `sit`) cộng 1 tương tác |
| **4. Kích thước ảnh** (rộng × cao, pixel) | Cố định **128 × 128** (bản Meta-Album đã thu nhỏ) | min 164 × 120<br>trung bình 457 × 397<br>max 500 × 500 | min 122 × 89<br>trung bình 592 × 496<br>max 640 × 640<br>(đo trên 3.000 ảnh) | min 128 × 145<br>trung bình 578 × 482<br>max 640 × 640 |
| **5. Có bbox người / vật?** | Bản OpenML (notebook EDA): **không có bbox nào**, nhãn gắn cho cả bức ảnh<br>*(bản gốc: có **bbox người** trong file XML, không có bbox vật; `data_preparation` dùng bản gốc này)* | **Chỉ bbox người**: mỗi người có nhãn đều có 1 bbox, không có bbox vật | **Cả hai**: mỗi cặp HOI có 1 bbox người + 1 bbox vật | **Cả hai**: bbox người, và bbox vật mà người đó tương tác (lấy từ annotation COCO). Hành động không cần vật (`stand`, `run`…) thì không có bbox vật |

## Giải thích thuật ngữ

| Thuật ngữ | Nghĩa | Ví dụ |
|---|---|---|
| **Nhãn** (label) | Tên hành động được gán cho ảnh hoặc cho người | `riding_a_bike` |
| **Đơn nhãn** (single-label) | Mỗi mẫu có **đúng 1** nhãn | Stanford: ảnh người đạp xe chỉ mang nhãn `riding_a_bike` |
| **Đa nhãn** (multi-label) | Một mẫu có thể mang **nhiều** nhãn cùng lúc | V-COCO: 1 người vừa `sit`, vừa `hold`, vừa `drink` |
| **bbox** (bounding box) | Khung chữ nhật đánh dấu vị trí người hoặc vật trong ảnh, gồm toạ độ góc trên-trái và góc dưới-phải | `(154, 71, 312, 395)`: x1 = 154, y1 = 71, x2 = 312, y2 = 395 |
| **HOI** (Human-Object Interaction) | Tương tác người–vật, ghi dạng cặp (vật, động từ) | `(horse, ride)` = cưỡi ngựa |
| **trainval / test** | Phần dữ liệu để huấn luyện và kiểm định / phần để đánh giá cuối | VOC chỉ công bố nhãn của `trainval` |

## Nhận xét nhanh

- **Đơn vị mẫu khác nhau:** Stanford tính theo **ảnh**, VOC và V-COCO tính theo **người**, HICO-DET tính theo **cặp người–vật**. Khi gộp 4 bộ cần đưa về cùng một đơn vị. Notebook đang chọn: 1 người (hoặc 1 ảnh) → 1 nhãn đích, dùng priority rule.
- **Ảnh Stanford rất nhỏ** (128 × 128) so với 3 bộ còn lại (khoảng 450–640 px). Phóng lên 224 × 224 cho CLIP thì ảnh bị mờ.
- **Chỉ HICO-DET và V-COCO có bbox vật**, nên chỉ 2 bộ này biết chính xác người đang tương tác với vật nào.
- **Số liệu HICO-DET trong notebook là mẫu 3.000 ảnh**, không phải toàn bộ bộ dữ liệu.

## Phụ lục: 40 nhãn của Stanford 40 Actions (số ảnh trong bản 3.389 ảnh)

`cutting_trees` 110 · `looking_through_a_microscope` 103 · `riding_a_bike` 99 · `smoking` 98 · `running` 98 · `cleaning_the_floor` 97 · `brushing_teeth` 97 · `phoning` 96 · `fishing` 95 · `jumping` 95 · `blowing_bubbles` 95 · `playing_guitar` 94 · `climbing` 94 · `throwing_frisby` 92 · `gardening` 92 · `riding_a_horse` 92 · `drinking` 92 · `pushing_a_cart` 91 · `playing_violin` 90 · `writing_on_a_book` 89 · `writing_on_a_board` 87 · `holding_an_umbrella` 87 · `cooking` 86 · `reading` 85 · `waving_hands` 85 · `fixing_a_bike` 84 · `shooting_an_arrow` 84 · `feeding_a_horse` 83 · `fixing_a_car` 80 · `taking_photos` 79 · `looking_through_a_telescope` 78 · `rowing_a_boat` 77 · `washing_dishes` 76 · `pouring_liquid` 76 · `applauding` 74 · `walking_the_dog` 73 · `cutting_vegetables` 71 · `texting_message` 60 · `using_a_computer` 44 · `watching_TV` 11

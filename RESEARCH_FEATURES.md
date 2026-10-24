1. Phát hiện cạnh

Sobel, Prewitt, Roberts: dùng đạo hàm ảnh để tìm biên.
Canny: bộ phát hiện cạnh phổ biến nhất, cho biên mảnh và ít nhiễu.
Laplacian / LoG (Laplacian of Gaussian).

2. Phát hiện góc và điểm đặc trưng

Harris, Shi-Tomasi: tìm góc.
FAST: phát hiện góc rất nhanh, phù hợp cho ứng dụng thời gian thực.

3. Mô tả cục bộ (keypoint + descriptor)

SIFT: bất biến với tỷ lệ và phép xoay, rất mạnh.
SURF: phiên bản nhanh hơn của SIFT.
ORB (FAST + BRIEF): nhanh và miễn phí bản quyền.
BRIEF, BRISK, FREAK: các descriptor nhị phân, gọn nhẹ.
KAZE / AKAZE.

4. Hình dạng và kết cấu

HOG (Histogram of Oriented Gradients): nổi tiếng trong phát hiện người đi bộ.
LBP (Local Binary Patterns): mô tả kết cấu, hay dùng cho nhận diện khuôn mặt.
Haar-like features: dùng trong bộ phát hiện khuôn mặt Viola–Jones.
Bộ lọc Gabor: phân tích kết cấu theo hướng và tần số.
GLCM (ma trận đồng xuất hiện mức xám): thống kê kết cấu.

5. Màu sắc và hình dạng toàn cục

Histogram màu, moment màu.
Hu moments, Zernike moments: mô tả hình dạng.
Hough transform: tìm đường thẳng và đường tròn.

6. Gộp đặc trưng cục bộ thành vector cho cả ảnh

Bag of Visual Words (BoVW).
VLAD, Fisher Vector.

Quy trình thường gặp là trích xuất đặc trưng (ví dụ HOG hoặc SIFT), rồi đưa vào một bộ phân loại như SVM hoặc Random Forest. Học sâu (CNN) đã thay thế phần lớn cách làm này vì mạng tự học được đặc trưng từ dữ liệu.
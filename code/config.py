"""Cấu hình chung cho toàn pipeline."""
import warnings
from pathlib import Path

# numpy 2.0 + Accelerate trên macOS báo nhầm "encountered in matmul" (kết quả vẫn hữu hạn)
warnings.filterwarnings("ignore", message=r".*encountered in matmul", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parent.parent
# Dữ liệu mặc định. 01_extract_features.py --labels đổi sang file nhãn khác (đường dẫn ảnh tính từ thư mục
# chứa file nhãn); file đã dùng được ghi vào data/features/<fXX>/run_info.json để notebook đọc lại đúng dữ liệu.
LABELS_CSV = ROOT / "data" / "vcoco_6class_crops" / "labels.csv"
DATA_DIR = LABELS_CSV.parent
# Mỗi lần chạy ghi vào 1 thư mục phiên bản: data/features/<version>/, data/results/<version>/ (xem versioning.py)
FEAT_ROOT = ROOT / "data" / "features"
RESULT_ROOT = ROOT / "data" / "results"

# Thứ tự lớp cố định -> label_id
CLASSES = ["hold", "ride", "eat/drink", "talk_on_phone", "work_on_computer", "unknown"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
SPLITS = ["train", "val", "test"]

# Đặc trưng cơ bản (yêu cầu ban đầu)
FEATURES = ["hog", "lbp", "color", "sift"]
# Đặc trưng bổ sung (khối nguyên tử lưu trong data/features/)
EXTRA_FEATURES = ["geom", "sift_spm", "hog_upper", "sift_upper", "gist", "skin"]
ATOMIC_FEATURES = FEATURES + EXTRA_FEATURES

# HOG
HOG_SIZE = (64, 128)          # (W, H)
# LBP
LBP_SIZE = (128, 256)
LBP_P, LBP_R = 24, 3
LBP_GRID = (3, 3)
# Color histogram (HSV)
COLOR_BINS = (8, 8, 8)
# SIFT + BoVW
SIFT_MAX_SIDE = 256
SIFT_STEP = 8
SIFT_SIZE = 16
BOVW_K = 500
BOVW_SAMPLE = 200_000

BOVW_SPM_WEIGHTS = (0.5, 0.5)   # trọng số mức 1x1 và 2x2 của spatial pyramid
# Vùng thân trên (đầu, tay gần mặt): UPPER_FRAC chiều cao phía trên crop
UPPER_FRAC = 0.4
HOG_UPPER_SIZE = (64, 64)
# GIST (Gabor filter bank)
GIST_SIZE = 128
GIST_SCALES = (5, 9, 15, 21)    # kích thước kernel ~ tần số
GIST_ORIENT = 8
GIST_GRID = 4
# Skin mask (YCrCb)
SKIN_CR = (133, 173)
SKIN_CB = (77, 127)
SKIN_GRID = 4

MIN_SIDE = 32                 # lọc crop quá nhỏ ở train (tùy chọn)
SEED = 42

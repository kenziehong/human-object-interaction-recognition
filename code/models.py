"""Định nghĩa 3 mô hình + lưới siêu tham số, và hàm nạp đặc trưng."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

import config as C

# Bộ đặc trưng dùng để train: tên -> danh sách khối nguyên tử được nối lại
FEATURE_SETS = {
    # 4 đặc trưng cơ bản
    "hog": ["hog"], "lbp": ["lbp"], "color": ["color"], "sift": ["sift"],
    # Đặc trưng bổ sung
    "geom": ["geom"],                              # hình học bbox người
    "sift_spm": ["sift_spm"],                      # SIFT-BoVW + spatial pyramid 1x1 + 2x2
    "upper": ["hog_upper", "sift_upper"],          # vùng thân trên (đầu / tay gần mặt)
    "gist": ["gist"],                              # Gabor / GIST: bối cảnh
    "skin": ["skin"],                              # vị trí vùng da
    # Phối hợp
    "fused": ["hog", "lbp", "color", "sift"],      # 4 cơ bản
    "combo_local": ["sift_spm", "sift_upper", "color", "geom", "skin"],   # vật nhỏ + vị trí + bối cảnh
    "fused_all": ["hog", "lbp", "color", "sift_spm", "hog_upper", "sift_upper", "gist", "skin", "geom"],
}

MODELS = ["logreg", "svm", "rf", "xgb"]
PCA_MIN_DIM = 1500               # bộ đặc trưng lớn hơn thì PCA 256 trước SVM


def blocks(feat):
    return FEATURE_SETS[feat]


def load_xy(feat_dir, split, feat):
    y = np.load(feat_dir / f"y_{split}.npy")
    X = np.hstack([np.load(feat_dir / f"{split}_{b}.npy") for b in blocks(feat)])
    return X.astype(np.float32), y


def make_model(name, feat, n_train, dim):
    if name == "logreg":
        pipe = Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))])
        grid = {"clf__C": [0.01, 0.1, 1, 10]}
    elif name == "svm":
        steps = [("scaler", StandardScaler())]
        if dim > PCA_MIN_DIM:
            steps.append(("pca", PCA(n_components=min(256, n_train - 1), random_state=C.SEED)))
        steps.append(("clf", SVC(kernel="rbf", class_weight="balanced")))
        pipe = Pipeline(steps)
        grid = {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", 1e-3, 1e-4]}
    elif name == "rf":
        pipe = Pipeline([("clf", RandomForestClassifier(class_weight="balanced_subsample",
                                                        n_jobs=1, random_state=C.SEED))])
        grid = {"clf__n_estimators": [300, 500], "clf__max_depth": [None, 30],
                "clf__min_samples_leaf": [1, 3]}
    elif name == "xgb":
        # XGBoost không có class_weight -> cân bằng bằng sample_weight (xem fit_params)
        # Chi phí XGBoost tỉ lệ với số cột -> PCA khi bộ đặc trưng lớn, max_bin=64 cho nhanh
        steps = []
        if dim > PCA_MIN_DIM:
            steps.append(("pca", PCA(n_components=min(256, n_train - 1), random_state=C.SEED)))
        steps.append(("clf", XGBClassifier(objective="multi:softprob", tree_method="hist", max_bin=64,
                                           subsample=0.8, colsample_bytree=0.5,
                                           n_jobs=1, random_state=C.SEED, verbosity=0)))
        pipe = Pipeline(steps)
        grid = {"clf__n_estimators": [300, 600], "clf__max_depth": [4, 6],
                "clf__learning_rate": [0.05, 0.1]}
    else:
        raise ValueError(name)
    return pipe, grid


def rank_results(df):
    """Xếp hạng theo val macro-F1; khi hoà thì xét balanced acc rồi tên, để kết quả lặp lại được."""
    return df.sort_values(["val_macro_f1", "val_balanced_acc", "feature", "model"],
                          ascending=[False, False, True, True], kind="mergesort")


def fit_params(name, y):
    """Tham số truyền vào .fit(): XGBoost cần sample_weight cân bằng lớp."""
    if name == "xgb":
        return {"clf__sample_weight": compute_sample_weight("balanced", y)}
    return {}


def use_all_cores(pipe, name):
    """Khi fit 1 mô hình đơn lẻ (không trong GridSearch) thì cho dùng hết nhân CPU."""
    if name in ("rf", "xgb"):
        pipe.set_params(clf__n_jobs=-1)
    return pipe

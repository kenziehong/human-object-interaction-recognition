"""Định nghĩa 4 mô hình + zero-shot CLIP, lưới siêu tham số, hàm nạp đặc trưng và hàm tính chỉ số đánh giá."""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_recall_fscore_support
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
    # Mốc so sánh (nhóm 'clip', không phải đặc trưng thủ công): embedding ảnh CLIP, chỉ có khi trích với --clip
    "clip": ["clip"],
}

MODELS = ["logreg", "svm", "rf", "xgb", "clip_zs"]   # clip_zs chỉ chạy với bộ đặc trưng "clip"
CLIP_ZS_T = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9]           # lưới ngưỡng T của clip_zs (0 = không gán unknown)
PCA_MIN_DIM = 1500               # bộ đặc trưng lớn hơn thì PCA 256 trước SVM


def blocks(feat):
    return FEATURE_SETS[feat]


def feature_group(feat):
    """Nhóm phương pháp: 'clip' (embedding CLIP, chỉ là mốc so sánh) hay 'handcrafted' (đặc trưng thủ công).
    Cấu hình tốt nhất trên val được chọn riêng trong từng nhóm."""
    return "clip" if "clip" in FEATURE_SETS.get(feat, [feat]) else "handcrafted"


def model_applies(name, feat):
    """clip_zs cần đúng embedding CLIP (so với embedding prompt), nên chỉ chạy với bộ đặc trưng chỉ gồm khối clip."""
    return name != "clip_zs" or blocks(feat) == ["clip"]


def available_feature_sets(feat_dir):
    """Các bộ đặc trưng có đủ khối trong phiên bản đặc trưng này (khối clip chỉ có khi trích với --clip)."""
    return [f for f in FEATURE_SETS if all((feat_dir / f"train_{b}.npy").exists() for b in blocks(f))]


class ClipZeroShot(ClassifierMixin, BaseEstimator):
    """Zero-shot CLIP: xác suất = softmax(CLIP_LOGIT_SCALE × cosine(ảnh, prompt)) trên các lớp hành động; lấy lớp
    cao nhất, nếu xác suất cao nhất < T thì gán unknown. Không học gì từ dữ liệu; T chọn trên val như mọi siêu tham số.

    Ngưỡng T mô phỏng lỏng ý tưởng ngưỡng chọn Reliable Person Query trong DP-HOI: ở bài báo T = 0.9 là ngưỡng điểm
    PHÁT HIỆN NGƯỜI, ở đây T là ngưỡng TIN CẬY HÀNH ĐỘNG để quyết định gán unknown."""

    def __init__(self, text_path=None, T=0.0):
        self.text_path = text_path
        self.T = T

    def fit(self, X, y=None, **fit_kw):
        self.text_ = np.load(self.text_path)
        self.action_ids_ = np.array([C.CLASS_TO_ID[c] for c in C.CLASSES if c in C.CLIP_PROMPTS])
        self.classes_ = np.arange(len(C.CLASSES))
        return self

    def action_proba(self, X):
        z = C.CLIP_LOGIT_SCALE * (X @ self.text_.T)
        z = np.exp(z - z.max(axis=1, keepdims=True))
        return z / z.sum(axis=1, keepdims=True)

    def predict(self, X):
        p = self.action_proba(X)
        pred = self.action_ids_[p.argmax(axis=1)]
        pred[p.max(axis=1) < self.T] = C.CLASS_TO_ID["unknown"]
        return pred


def load_xy(feat_dir, split, feat):
    y = np.load(feat_dir / f"y_{split}.npy")
    X = np.hstack([np.load(feat_dir / f"{split}_{b}.npy") for b in blocks(feat)])
    return X.astype(np.float32), y


def make_model(name, feat, n_train, dim, feat_dir=None):
    if name == "clip_zs":
        # feat_dir chứa clip_text.npy (embedding prompt, trích ở bước 1 cùng khối clip)
        pipe = Pipeline([("clf", ClipZeroShot(text_path=str(feat_dir / "clip_text.npy") if feat_dir else None))])
        grid = {"clf__T": CLIP_ZS_T}
    elif name == "logreg":
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


def eval_metrics(y, pred, prefix):
    """Chỉ số đánh giá của 1 cấu hình trên 1 split. Macro-F1 là chỉ số chính; accuracy chỉ để tham khảo
    (lớp unknown chiếm đa số). Macro recall = balanced accuracy nên không ghi riêng."""
    labels = list(range(len(C.CLASSES)))
    p, r, f, _ = precision_recall_fscore_support(y, pred, labels=labels, average=None, zero_division=0)
    return {f"{prefix}_macro_f1": f1_score(y, pred, labels=labels, average="macro", zero_division=0),
            f"{prefix}_balanced_acc": balanced_accuracy_score(y, pred),
            f"{prefix}_accuracy": accuracy_score(y, pred),
            f"{prefix}_macro_precision": p.mean(),
            f"{prefix}_weighted_f1": f1_score(y, pred, labels=labels, average="weighted", zero_division=0),
            **{f"recall_{c}": v for c, v in zip(C.CLASSES, r)},
            **{f"precision_{c}": v for c, v in zip(C.CLASSES, p)},
            **{f"f1_{c}": v for c, v in zip(C.CLASSES, f)}}


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

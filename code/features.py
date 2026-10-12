"""4 loại đặc trưng thủ công: HOG, LBP, Color Histogram, SIFT (Bag of Visual Words)."""
import cv2
import numpy as np
from skimage.feature import hog, local_binary_pattern

import config as C


def _l1(v):
    return v / (v.sum() + 1e-8)


# ---------- HOG: hình dáng / tư thế ----------
def hog_feat(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, C.HOG_SIZE, interpolation=cv2.INTER_AREA)
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


# ---------- LBP: kết cấu, histogram trên lưới 3x3 ----------
def lbp_image(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, C.LBP_SIZE, interpolation=cv2.INTER_AREA)
    return local_binary_pattern(gray, C.LBP_P, C.LBP_R, method="uniform")


def lbp_feat(bgr):
    lbp = lbp_image(bgr)
    n_bins = C.LBP_P + 2
    gh, gw = C.LBP_GRID
    H, W = lbp.shape
    hists = []
    for i in range(gh):
        for j in range(gw):
            cell = lbp[i * H // gh:(i + 1) * H // gh, j * W // gw:(j + 1) * W // gw]
            h, _ = np.histogram(cell, bins=n_bins, range=(0, n_bins))
            hists.append(_l1(h.astype(np.float32)))
    return np.concatenate(hists)


# ---------- Color histogram HSV 8x8x8, nửa trên + nửa dưới ----------
def color_feat(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H = hsv.shape[0]
    hists = []
    for part in (hsv[: H // 2], hsv[H // 2:]):
        h = cv2.calcHist([part], [0, 1, 2], None, list(C.COLOR_BINS), [0, 180, 0, 256, 0, 256])
        hists.append(_l1(h.flatten().astype(np.float32)))
    return np.concatenate(hists)


# ---------- SIFT dày đặc + Bag of Visual Words ----------
_sift = None


def sift_descriptors(bgr):
    """Dense SIFT: lưới keypoint bước SIFT_STEP, trả về (n, 128) float32."""
    return sift_desc_pos(bgr)[0]


def sift_desc_pos(bgr):
    """Dense SIFT kèm vị trí keypoint chuẩn hóa về [0, 1): trả về (desc (n,128), pos (n,2) = (x, y))."""
    global _sift
    if _sift is None:
        _sift = cv2.SIFT_create()
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    s = C.SIFT_MAX_SIDE / max(h, w)
    gray = cv2.resize(gray, (max(int(round(w * s)), C.SIFT_SIZE), max(int(round(h * s)), C.SIFT_SIZE)),
                      interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    h, w = gray.shape
    half = C.SIFT_SIZE // 2
    kps = [cv2.KeyPoint(float(x), float(y), C.SIFT_SIZE)
           for y in range(half, max(h - half, half + 1), C.SIFT_STEP)
           for x in range(half, max(w - half, half + 1), C.SIFT_STEP)]
    kps, desc = _sift.compute(gray, kps)
    if desc is None:
        return np.zeros((0, 128), np.float32), np.zeros((0, 2), np.float32)
    pos = np.array([[k.pt[0] / w, k.pt[1] / h] for k in kps], np.float32)
    return desc.astype(np.float32), pos


def bovw_hist(desc, kmeans):
    """Histogram từ vựng, chuẩn hóa L1 rồi căn bậc 2 (Hellinger)."""
    k = kmeans.n_clusters
    if len(desc) == 0:
        return np.zeros(k, np.float32)
    words = kmeans.predict(desc)
    h = np.bincount(words, minlength=k).astype(np.float32)
    return np.sqrt(_l1(h))


def _hist_words(words, k):
    return np.sqrt(_l1(np.bincount(words, minlength=k).astype(np.float32)))


def bovw_all(desc, pos, kmeans):
    """Từ 1 lần predict: BoVW toàn ảnh, Spatial Pyramid (1x1 + 2x2), BoVW vùng thân trên."""
    k = kmeans.n_clusters
    if len(desc) == 0:
        z = np.zeros(k, np.float32)
        return {"sift": z, "sift_spm": np.zeros(5 * k, np.float32), "sift_upper": z}
    words = kmeans.predict(desc)
    w0, w1 = C.BOVW_SPM_WEIGHTS
    full = _hist_words(words, k)
    cells = []
    for gy in range(2):
        for gx in range(2):
            m = ((pos[:, 0] >= gx / 2) & (pos[:, 0] < (gx + 1) / 2) &
                 (pos[:, 1] >= gy / 2) & (pos[:, 1] < (gy + 1) / 2))
            cells.append(_hist_words(words[m], k) * w1)
    upper = _hist_words(words[pos[:, 1] < C.UPPER_FRAC], k)
    return {"sift": full, "sift_spm": np.concatenate([full * w0] + cells), "sift_upper": upper}


# ---------- HOG vùng thân trên: nơi diễn ra gọi điện / ăn uống ----------
def hog_upper_feat(bgr):
    top = bgr[: max(1, int(round(bgr.shape[0] * C.UPPER_FRAC)))]
    gray = cv2.resize(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY), C.HOG_UPPER_SIZE, interpolation=cv2.INTER_AREA)
    return hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
               block_norm="L2-Hys", feature_vector=True).astype(np.float32)


# ---------- GIST: năng lượng Gabor (nhiều tỉ lệ x hướng) trên lưới 4x4, mô tả bối cảnh ----------
_gabor = None


def gabor_bank():
    global _gabor
    if _gabor is None:
        _gabor = []
        for ks in C.GIST_SCALES:
            for i in range(C.GIST_ORIENT):
                kern = cv2.getGaborKernel((ks, ks), sigma=ks / 4, theta=np.pi * i / C.GIST_ORIENT,
                                          lambd=ks / 2, gamma=0.5, psi=0)
                _gabor.append((kern - kern.mean()).astype(np.float32))
    return _gabor


def gist_feat(bgr):
    gray = cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), (C.GIST_SIZE, C.GIST_SIZE),
                      interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    gray = (gray - gray.mean()) / (gray.std() + 1e-6)
    g, n = C.GIST_GRID, C.GIST_SIZE // C.GIST_GRID
    out = []
    for kern in gabor_bank():
        resp = np.abs(cv2.filter2D(gray, cv2.CV_32F, kern))
        out.append(resp.reshape(g, n, g, n).mean(axis=(1, 3)).ravel())
    v = np.concatenate(out)
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


# ---------- Skin mask (YCrCb): vị trí tay / mặt ----------
def skin_mask(bgr):
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[..., 1], ycc[..., 2]
    return ((cr >= C.SKIN_CR[0]) & (cr <= C.SKIN_CR[1]) & (cb >= C.SKIN_CB[0]) & (cb <= C.SKIN_CB[1]))


def skin_feat(bgr):
    m = cv2.resize(skin_mask(bgr).astype(np.float32), (64, 128), interpolation=cv2.INTER_AREA)
    g = C.SKIN_GRID
    grid = m.reshape(g, 128 // g, g, 64 // g).mean(axis=(1, 3)).ravel()       # tỉ lệ da trong từng ô
    ys, xs = np.nonzero(m > 0.5)
    if len(ys):
        stats = [m.mean(), ys.mean() / 128, xs.mean() / 64, ys.std() / 128, xs.std() / 64]
    else:
        stats = [0, 0, 0, 0, 0]
    return np.concatenate([grid, np.array(stats, np.float32)]).astype(np.float32)


# ---------- Hình học từ bbox (không cần ảnh) ----------
def _box(s):
    return np.array([float(v) for v in s.split(",")])


def geom_feat(row):
    """Tỉ lệ khung người, kích thước, và mức nới thực tế mỗi phía (< 20% nghĩa là người sát mép ảnh)."""
    px1, py1, px2, py2 = _box(row["person_bbox"])
    cx1, cy1, cx2, cy2 = _box(row["crop_bbox"])
    pw, ph = max(px2 - px1, 1.0), max(py2 - py1, 1.0)
    return np.array([np.log(pw / ph), np.log(pw), np.log(ph), np.log(pw * ph),
                     (px1 - cx1) / pw, (cx2 - px2) / pw, (py1 - cy1) / ph, (cy2 - py2) / ph], np.float32)


SIMPLE_EXTRACTORS = {"hog": hog_feat, "lbp": lbp_feat, "color": color_feat,
                     "hog_upper": hog_upper_feat, "gist": gist_feat, "skin": skin_feat}

"""Bước 1-2: trích đặc trưng (4 cơ bản + bổ sung) cho train/val/test, cache vào data/features/.

Chạy:  python 01_extract_features.py --version f01            (ghi vào data/features/f01/)
       python 01_extract_features.py --version f_test --limit 60 (thử nhanh)
       python 01_extract_features.py --version f02 --labels data/train_test_split/labels_split_unk3000.csv \
              --data_dir data/dataset_6class_crops
       thêm --clip để trích cả khối CLIP (cần torch + open_clip_torch; mốc so sánh, xem clip_feat.py)
Không ghi đè phiên bản đã có.
"""
import argparse
import os
import platform
import time

import pandas as pd

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.cluster import MiniBatchKMeans

import config as C
import resources as R
from data_utils import load_labels, read_image, set_labels_csv
import versioning as V
from features import SIMPLE_EXTRACTORS, bovw_all, geom_feat, sift_desc_pos, sift_descriptors


def _chunks(seq, n):
    return [seq[i:i + n] for i in range(0, len(seq), n)]


def _simple_chunk(paths):
    """Trả về (đặc trưng, thời gian CPU từng đặc trưng tính bằng giây)."""
    out = {k: [] for k in SIMPLE_EXTRACTORS}
    secs = {k: 0.0 for k in ["read_image", *SIMPLE_EXTRACTORS]}
    for p in paths:
        t = time.perf_counter()
        img = read_image(p)
        secs["read_image"] += time.perf_counter() - t
        for k, f in SIMPLE_EXTRACTORS.items():
            t = time.perf_counter()
            out[k].append(f(img))
            secs[k] += time.perf_counter() - t
    return {k: np.stack(v) for k, v in out.items()}, secs


def _sift_sample_chunk(paths, per_img, seed):
    rng = np.random.default_rng(seed)
    descs = []
    for p in paths:
        d = sift_descriptors(read_image(p))
        if len(d) > per_img:
            d = d[rng.choice(len(d), per_img, replace=False)]
        descs.append(d)
    return np.concatenate(descs)


SIFT_OUTPUTS = ["sift", "sift_spm", "sift_upper"]


def _bovw_chunk(paths, kmeans):
    hs, secs = [], {"read_image": 0.0, "sift_descriptor": 0.0, "bovw_encode": 0.0}
    for p in paths:
        t0 = time.perf_counter()
        img = read_image(p)
        t1 = time.perf_counter()
        desc, pos = sift_desc_pos(img)
        t2 = time.perf_counter()
        hs.append(bovw_all(desc, pos, kmeans))
        t3 = time.perf_counter()
        secs["read_image"] += t1 - t0
        secs["sift_descriptor"] += t2 - t1
        secs["bovw_encode"] += t3 - t2
    return {k: np.stack([h[k] for h in hs]) for k in SIFT_OUTPUTS}, secs


def _sum_secs(parts):
    tot = {}
    for _, secs in parts:
        for k, v in secs.items():
            tot[k] = tot.get(k, 0.0) + v
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="tên phiên bản đặc trưng, ví dụ f01 -> data/features/f01/")
    ap.add_argument("--note", default="", help="mô tả ngắn phiên bản (ghi vào run_info.json)")
    ap.add_argument("--limit", type=int, default=None, help="chỉ lấy N ảnh mỗi split (thử nhanh)")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--labels", default=None,
                    help="file nhãn có cột split (tương đối với thư mục gốc repo); mặc định config.LABELS_CSV")
    ap.add_argument("--data_dir", default=None, help="thư mục ảnh mà crop_path tính từ đó; mặc định cạnh --labels")
    ap.add_argument("--clip", action="store_true", help="trích thêm khối clip (embedding ảnh CLIP) và clip_text.npy")
    args = ap.parse_args()
    if args.labels:
        set_labels_csv(args.labels, args.data_dir)
    labels_csv, data_dir = (str(p.relative_to(C.ROOT)) for p in (C.LABELS_CSV, C.DATA_DIR))
    print("Dữ liệu:", labels_csv, "| ảnh:", data_dir)

    out_dir = V.prepare_dir(V.feat_dir(args.version), "y_train.npy")
    res_dir = out_dir           # thời gian trích đặc trưng lưu cùng phiên bản đặc trưng
    par = Parallel(n_jobs=args.n_jobs, verbose=0)
    t_start = time.time()
    mon = R.ResourceMonitor().start()
    m_all = mon.mark()
    timing = []   # wall_s = thời gian thực của cả bước (song song); cpu_s = tổng thời gian tính của riêng khối đó

    def log_time(stage, split, block, n, wall_s, cpu_s, res=None):
        # res = tài nguyên của cả công đoạn (mọi khối của công đoạn đó có cùng giá trị):
        # stage_peak_rss_mb = RAM đỉnh cả cây tiến trình, stage_cpu_util = số nhân dùng trung bình
        timing.append({"stage": stage, "split": split, "block": block, "n_images": n,
                       "wall_s": round(wall_s, 2), "cpu_s": round(cpu_s, 2),
                       "cpu_ms_per_img": round(1000 * cpu_s / max(n, 1), 2),
                       "stage_peak_rss_mb": res and res["peak_rss_mb"], "stage_peak_delta_mb": res and res["peak_delta_mb"],
                       "stage_cpu_util": res and res["cpu_util"]})

    dfs = {}
    for split in C.SPLITS:
        df = load_labels(split)
        if args.limit:
            df = df.groupby("label_id", group_keys=False).head(max(1, args.limit // 6)).reset_index(drop=True)
        dfs[split] = df
        np.save(out_dir / f"y_{split}.npy", df["label_id"].to_numpy())
        # Cột định danh khác nhau giữa các bộ dữ liệu (V-COCO: image_id/ann_id; bộ gộp: dataset/source_image)
        meta_cols = [c for c in ["crop_path", "label", "label_id", "dataset", "source_image", "image_id", "ann_id",
                                 "crop_w", "crop_h"] if c in df]
        df[meta_cols].to_csv(out_dir / f"meta_{split}.csv", index=False)

    # Hình học từ bbox (không cần đọc ảnh)
    for split, df in dfs.items():
        m = mon.mark()
        t = time.perf_counter()
        X = np.stack([geom_feat(r) for _, r in df.iterrows()])
        dt = time.perf_counter() - t
        log_time("extract", split, "geom", len(df), dt, dt, mon.since(m))
        np.save(out_dir / f"{split}_geom.npy", X)
        print(f"[{split}] geom  {X.shape}")

    # Các đặc trưng tính trực tiếp từ ảnh: HOG, LBP, Color, HOG thân trên, GIST, Skin
    for split, df in dfs.items():
        m = mon.mark()
        t = time.time()
        parts = par(delayed(_simple_chunk)(c) for c in _chunks(df["path"].tolist(), 64))
        wall = time.time() - t
        for k in SIMPLE_EXTRACTORS:
            X = np.concatenate([p[0][k] for p in parts])
            np.save(out_dir / f"{split}_{k}.npy", X)
            print(f"[{split}] {k:5s} {X.shape}")
        res = mon.since(m)
        for k, v in _sum_secs(parts).items():
            log_time("extract_image (hog/lbp/color/hog_upper/gist/skin)", split, k, len(df), wall, v, res)
        print(f"[{split}] đặc trưng ảnh xong trong {wall:.0f}s")

    # SIFT BoVW: codebook chỉ học trên train
    m = mon.mark()
    t = time.time()
    train_paths = dfs["train"]["path"].tolist()
    per_img = int(np.ceil(C.BOVW_SAMPLE / len(train_paths)))
    samples = par(delayed(_sift_sample_chunk)(c, per_img, C.SEED + i)
                  for i, c in enumerate(_chunks(train_paths, 64)))
    D = np.concatenate(samples)
    print(f"Descriptor SIFT lấy mẫu từ train: {D.shape}")
    k = min(C.BOVW_K, len(D))
    kmeans = MiniBatchKMeans(n_clusters=k, batch_size=4096, n_init=3, random_state=C.SEED).fit(D)
    joblib.dump(kmeans, out_dir / "sift_kmeans.joblib")
    log_time("sift_codebook (lấy mẫu descriptor + KMeans)", "train", f"kmeans_K{k}", len(train_paths),
             time.time() - t, time.time() - t, mon.since(m))
    print(f"KMeans K={k} xong trong {time.time() - t:.0f}s")

    for split, df in dfs.items():
        m = mon.mark()
        t = time.time()
        parts = par(delayed(_bovw_chunk)(c, kmeans) for c in _chunks(df["path"].tolist(), 64))
        wall = time.time() - t
        for k in SIFT_OUTPUTS:
            X = np.concatenate([p[0][k] for p in parts])
            np.save(out_dir / f"{split}_{k}.npy", X)
            print(f"[{split}] {k:10s} {X.shape}")
        res = mon.since(m)
        for k, v in _sum_secs(parts).items():
            log_time("sift_bovw (sift/sift_spm/sift_upper)", split, k, len(df), wall, v, res)
        print(f"[{split}] BoVW xong trong {wall:.0f}s")

    # CLIP: embedding ảnh (không học gì từ dữ liệu) + embedding prompt cho zero-shot; chạy tuần tự trên CPU
    feats = C.ATOMIC_FEATURES + (C.CLIP_FEATURES if args.clip else [])
    if args.clip:
        import clip_feat
        np.save(out_dir / "clip_text.npy", clip_feat.text_embeddings())
        for split, df in dfs.items():
            m = mon.mark()
            t = time.time()
            X = clip_feat.image_embeddings(df["path"].tolist())
            wall = time.time() - t
            np.save(out_dir / f"{split}_clip.npy", X)
            log_time(f"clip ({C.CLIP_MODEL} {C.CLIP_PRETRAINED})", split, "clip", len(df), wall, wall, mon.since(m))
            print(f"[{split}] clip {X.shape} trong {wall:.0f}s")

    # Kiểm tra
    for split in C.SPLITS:
        n = len(dfs[split])
        for f in feats:
            X = np.load(out_dir / f"{split}_{f}.npy")
            assert len(X) == n and np.isfinite(X).all(), (split, f)
    print("OK: đủ số dòng, không NaN/Inf ->", out_dir)

    tdf = pd.DataFrame(timing)
    tdf["machine"] = f"{platform.machine()} {os.cpu_count()} CPU, n_jobs={args.n_jobs}"
    tdf.to_csv(res_dir / "timing_extract.csv", index=False)
    print(f"Thời gian trích đặc trưng -> {res_dir / 'timing_extract.csv'}")
    res = mon.since(m_all)
    mon.stop()
    disk = {f: round(sum(R.file_mb(out_dir / f"{s}_{f}.npy") for s in C.SPLITS), 1) for f in feats}
    V.write_run_info(out_dir, "01_extract_features", args, note=args.note, labels_csv=labels_csv, data_dir=data_dir,
                     n_images={s: len(d) for s, d in dfs.items()}, total_wall_s=round(time.time() - t_start, 1),
                     peak_rss_mb=res["peak_rss_mb"], peak_delta_mb=res["peak_delta_mb"], cpu_s=res["cpu_s"], cpu_util=res["cpu_util"],
                     disk_mb=R.dir_mb(out_dir), disk_mb_by_block=disk, blocks=feats,
                     feature_config=V.feature_config())
    print(f"RAM đỉnh {res['peak_rss_mb']} MB, CPU {res['cpu_s']} s ({res['cpu_util']} nhân), "
          f"đĩa {R.dir_mb(out_dir)} MB")
    print(tdf.groupby("block")[["cpu_s"]].sum().assign(
        cpu_ms_per_img=lambda d: (1000 * d.cpu_s / sum(len(v) for v in dfs.values())).round(2)).to_string())


if __name__ == "__main__":
    main()

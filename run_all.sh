#!/usr/bin/env bash
# Chạy toàn bộ pipeline theo thứ tự: trích đặc trưng -> chọn mô hình trên val -> test -> notebook báo cáo.
# Cách dùng (ở thư mục gốc repo):
#   bash run_all.sh                     # tạo data/features/f01 và data/results/r01
#   bash run_all.sh f02 r02 "ghi chú"   # phiên bản khác
# Tuỳ chọn qua biến môi trường:
#   LABELS=data/train_test_split/labels_split_unk3000.csv       file nhãn (mặc định trong config.py)
#   DATA_DIR=data/dataset_6class_crops                          thư mục ảnh (mặc định: cạnh file nhãn)
#   FEATS="sift hog"  MODELS="logreg"                            chỉ chạy một phần (mặc định: tất cả)
#   CLIP=1                                                       bước 1 trích thêm khối clip (cần torch, open_clip_torch)
# Script dừng nếu phiên bản đã tồn tại (không ghi đè kết quả cũ).
set -euo pipefail

FEAT=${1:-f01}
RES=${2:-r01}
NOTE=${3:-"chạy toàn bộ pipeline"}
LABELS=${LABELS:-}
DATA_DIR=${DATA_DIR:-}
FEATS=${FEATS:-}
MODELS=${MODELS:-}
CLIP=${CLIP:-}

cd "$(dirname "$0")/code"
PY=../.venv/bin/python
[ -x "$PY" ] || { echo "Chưa có .venv: xem mục Cài đặt trong README"; exit 1; }

echo "== Bước 1: trích đặc trưng -> data/features/$FEAT"
if [ -e "../data/features/$FEAT/y_train.npy" ]; then
  echo "   data/features/$FEAT đã có, dùng lại"
else
  $PY 01_extract_features.py --version "$FEAT" --note "$NOTE" ${LABELS:+--labels "$LABELS"} ${DATA_DIR:+--data_dir "$DATA_DIR"} ${CLIP:+--clip}
fi

echo "== Bước 2: chọn mô hình trên val -> data/results/$RES"
# shellcheck disable=SC2086  (FEATS/MODELS cố ý tách theo khoảng trắng)
$PY 02_train_select.py --feat_version "$FEAT" --version "$RES" --note "$NOTE" \
  ${FEATS:+--feats $FEATS} ${MODELS:+--models $MODELS}

echo "== Bước 3: đánh giá test"
$PY 03_final_test.py --version "$RES"

echo "== Bước 4: chạy notebook báo cáo (hiển thị $RES)"
RUN_VERSION="$RES" ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 04_report.ipynb

echo "Xong. Xem kết quả trong code/04_report.ipynb và data/results/$RES/"

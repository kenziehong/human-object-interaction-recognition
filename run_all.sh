#!/usr/bin/env bash
# Chạy toàn bộ pipeline theo thứ tự: trích đặc trưng -> chọn mô hình trên val -> test -> notebook báo cáo.
# Cách dùng (ở thư mục gốc repo):
#   bash run_all.sh                     # tạo features/f01 và results/r01
#   bash run_all.sh f02 r02 "ghi chú"   # phiên bản khác
# Script dừng nếu phiên bản đã tồn tại (không ghi đè kết quả cũ).
set -euo pipefail

FEAT=${1:-f01}
RES=${2:-r01}
NOTE=${3:-"chạy toàn bộ pipeline"}

cd "$(dirname "$0")/code"
PY=../.venv/bin/python
[ -x "$PY" ] || { echo "Chưa có .venv: xem mục Cài đặt trong README"; exit 1; }

echo "== Bước 1: trích đặc trưng -> features/$FEAT"
if [ -e "../features/$FEAT/y_train.npy" ]; then
  echo "   features/$FEAT đã có, dùng lại"
else
  $PY 01_extract_features.py --version "$FEAT" --note "$NOTE"
fi

echo "== Bước 2: chọn mô hình trên val -> results/$RES"
$PY 02_train_select.py --feat_version "$FEAT" --version "$RES" --note "$NOTE"

echo "== Bước 3: đánh giá test"
$PY 03_final_test.py --version "$RES"

echo "== Bước 4: chạy notebook báo cáo (hiển thị $RES)"
RUN_VERSION="$RES" ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace 04_report.ipynb

echo "Xong. Xem kết quả trong code/04_report.ipynb và results/$RES/"

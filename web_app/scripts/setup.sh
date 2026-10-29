#!/usr/bin/env bash
# Tải mã nguồn DP-HOI (CVPR 2024) và checkpoint pre-train ResNet50 vào đúng chỗ web_app cần.
set -e
cd "$(dirname "$0")/.."
[ -d third_party/DP-HOI ] || git clone --depth 1 https://github.com/xingaoli/DP-HOI.git third_party/DP-HOI
[ -f weights/dphoi_res50_3layers.pth ] || gdown 1DYBY8wHDGDPh9pyvxBewTqmpbztAzazb -O weights/dphoi_res50_3layers.pth
echo "Xong. Chạy: python3 app.py"

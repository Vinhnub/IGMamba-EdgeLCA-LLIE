#!/bin/bash
# ==============================================================================
# Script cài đặt môi trường EdgeLLIE trên NVIDIA Jetson Orin Nano
# Hỗ trợ JetPack 5.1 / 6.0 / 6.1 (Ubuntu 20.04 / 22.04 LTS)
# ==============================================================================

set -e

echo "=========================================================="
echo "   CÀI ĐẶT EDGELLIE TRÊN NVIDIA JETSON ORIN NANO"
echo "=========================================================="

# 1. Cập nhật hệ thống & cài các gói cần thiết
echo "[1/5] Cập nhật các gói hệ thống..."
sudo apt update
sudo apt install -y python3-pip python3-dev libopenblas-dev libjpeg-dev zlib1g-dev v4l-utils

# 2. Cài đặt và cấu hình jetson-stats (jtop) để đọc GPU/CPU/VRAM/Temp
echo "[2/5] Đang cài đặt và kích hoạt jetson-stats (jtop)..."
sudo -H pip3 install -U jetson-stats

# Nạp lại systemd daemon và kích hoạt service jtop tự khởi động cùng hệ thống
sudo systemctl daemon-reload
sudo systemctl enable jtop.service
sudo systemctl restart jtop.service || true

# Cấp quyền cho user hiện tại vào group jtop và video/i2c (không cần sudo khi đọc telemetry)
if getent group jtop > /dev/null 2>&1; then
    sudo usermod -aG jtop $USER
fi
sudo usermod -aG video,i2c $USER || true

# 3. Cài đặt các thư viện Python cho Webapp
echo "[3/5] Đang cài đặt thư viện Python (FastAPI, Uvicorn, OpenCV)..."
pip3 install fastapi uvicorn pydantic psutil numpy websockets

# 4. Kiểm tra dịch vụ jtop
echo "[4/5] Kiểm tra trạng thái dịch vụ jtop..."
if systemctl is-active --quiet jtop.service; then
    echo "  -> jtop.service: ĐANG HOẠT ĐỘNG (ACTIVE)"
else
    echo "  -> jtop.service: Đã cấu hình, sẽ kích hoạt sau khi khởi động lại."
fi
jtop --version || true

# 5. Cấu hình kiểm tra camera
echo "[5/5] Kiểm tra các cổng camera đang kết nối (CSI / USB):"
v4l2-ctl --list-devices || true

echo "=========================================================="
echo " HOÀN TẤT CÀI ĐẶT!"
echo " LƯU Ý QUAN TRỌNG:"
echo " 1. Hãy chạy: newgrp jtop (hoặc khởi động lại máy: sudo reboot)"
echo "    để quyền truy cập jtop có hiệu lực hoàn toàn."
echo " 2. Kiểm tra phần cứng bằng lệnh: jtop"
echo " 3. Khởi chạy ứng dụng: ./run_jetson.sh"
echo "=========================================================="

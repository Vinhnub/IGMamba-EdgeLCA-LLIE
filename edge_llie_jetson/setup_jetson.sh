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

# 2. Cài đặt và cấu hình jetson-stats (jtop) - TÙY CHỌN (Optional)
echo "[2/6] Đang cài đặt jetson-stats (Tùy chọn)..."
sudo -H pip3 install -U jetson-stats || true
sudo systemctl daemon-reload || true

if systemctl list-unit-files 2>/dev/null | grep -q "jetson_stats.service"; then
    echo "  -> Tìm thấy service: jetson_stats.service"
    sudo systemctl enable jetson_stats.service || true
    sudo systemctl restart jetson_stats.service || true
elif systemctl list-unit-files 2>/dev/null | grep -q "jtop.service"; then
    echo "  -> Tìm thấy service: jtop.service"
    sudo systemctl enable jtop.service || true
    sudo systemctl restart jtop.service || true
else
    echo "  [*] jtop service chưa có trong systemd (Hệ thống sẽ tự động dùng Tegra sysfs mặc định)."
fi

# Cấp quyền cho user hiện tại vào group jtop và video/i2c (không cần sudo khi đọc telemetry)
if getent group jtop > /dev/null 2>&1; then
    sudo usermod -aG jtop $USER
fi
sudo usermod -aG video,i2c $USER || true

# 3. Cài đặt các thư viện Python cho Webapp
echo "[3/6] Đang cài đặt thư viện Python (FastAPI, Uvicorn, OpenCV)..."
pip3 install fastapi uvicorn pydantic psutil numpy websockets torchvision

# 4. Kiểm tra PyTorch có hỗ trợ CUDA trên Jetson (ARM64)
echo "[4/6] Kiểm tra PyTorch & CUDA..."
python3 -c "
import torch
print('  -> PyTorch Version:', torch.__version__)
print('  -> CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('  -> Device Name:', torch.cuda.get_device_name(0))
    print('  -> Device Arch Capability:', torch.cuda.get_device_capability(0))
else:
    print('  [!] CẢNH BÁO: PyTorch chưa bật CUDA!')
    print('  [!] Trên Jetson, không thể cài torch bằng pip thông thường.')
    print('  [!] Hãy cài bản PyTorch chính thức từ NVIDIA Wheels:')
    print('      Xem chi tiết tại: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048')
" || true

# 5. Cấu hình biến môi trường kiến trúc GPU Orin Nano (Ampere sm_87)
echo "[5/6] Thiết lập biến môi trường GPU Ampere (sm_87)..."
export TORCH_CUDA_ARCH_LIST="8.7"
if ! grep -q "TORCH_CUDA_ARCH_LIST" ~/.bashrc; then
    echo 'export TORCH_CUDA_ARCH_LIST="8.7"' >> ~/.bashrc
fi

# 6. Kiểm tra dịch vụ jtop & camera
echo "[6/6] Kiểm tra trạng thái dịch vụ jtop & camera:"
if systemctl is-active --quiet jtop.service; then
    echo "  -> jtop.service: ĐANG HOẠT ĐỘNG (ACTIVE)"
else
    echo "  -> jtop.service: Đã cấu hình, sẽ kích hoạt sau khi khởi động lại."
fi
jtop --version || true
v4l2-ctl --list-devices || true

echo "=========================================================="
echo " HOÀN TẤT CÀI ĐẶT TRÊN JETSON ORIN NANO!"
echo " LƯU Ý QUAN TRỌNG:"
echo " 1. Hãy chạy: newgrp jtop (hoặc khởi động lại máy: sudo reboot)"
echo "    để quyền truy cập jtop có hiệu lực hoàn toàn."
echo " 2. Kiểm tra phần cứng bằng lệnh: jtop"
echo " 3. Khởi chạy ứng dụng: ./run_jetson.sh"
echo "=========================================================="

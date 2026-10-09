#!/bin/bash
# ==============================================================================
# Script khởi chạy tối ưu EdgeLLIE trên NVIDIA Jetson Orin Nano
# ==============================================================================

echo "=========================================================="
echo "   KHỞI ĐỘNG EDGELLIE TRÊN NVIDIA JETSON ORIN NANO"
echo "=========================================================="

# 1. Kích hoạt chế độ công suất tối đa 15W trên Orin Nano
# Chế độ 0 (15W 6-core) hoặc chế độ 1 (7W tiết kiệm điện)
if command -v nvpmodel &> /dev/null; then
    echo "[*] Thiết lập chế độ công suất 15W (MODE 0)..."
    sudo nvpmodel -m 0
fi

# 2. Khóa xung nhịp cao nhất cho GPU & CPU (Jetson Clocks)
if command -v jetson_clocks &> /dev/null; then
    echo "[*] Kích hoạt jetson_clocks để tối đa hóa hiệu năng GPU..."
    sudo jetson_clocks
fi

# 3. Lấy IP cục bộ của Jetson để tiện truy cập
JETSON_IP=$(hostname -I | awk '{print $1}')

echo "=========================================================="
echo " Đang khởi chạy web server..."
echo " Truy cập Dashboard tại: http://$JETSON_IP:8000"
echo " Hoặc từ trình duyệt cục bộ: http://localhost:8000"
echo "=========================================================="

# 4. Chạy server FastAPI
python3 -m edge_llie_jetson.app

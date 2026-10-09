# EdgeLLIE - Webapp Tăng Sáng Ảnh Triển Khai Trên NVIDIA Jetson Orin Nano

Hệ thống giao diện web giám sát và điều khiển tăng sáng ảnh thời gian thực (Low-Light Image Enhancement - LLIE) được thiết kế đặc thù cho nền tảng phần cứng **NVIDIA Jetson Orin Nano** (15W / 7W Power Mode).

Giao diện được xây dựng tương tự mẫu thiết bị Edge AI (Figure 8) với đầy đủ các khối chức năng, thông số phần cứng thực tế và **hỗ trợ lựa chọn đa dạng các mô hình trọng số (Pretrained Weights)**.

---

## 1. Cấu Trúc Giao Diện & Các Chỉ Số Phần Cứng

### Cột bên trái: Giám Sát Hình Ảnh & Tín Hiệu Ánh Sáng
- **CAMERA & LUMINANCE SENSOR**: Khung hiển thị luồng video MJPEG từ Camera CSI (GStreamer `nvarguscamerasrc`), Camera USB V4L2 hoặc các mẫu ảnh thiếu sáng từ tập dữ liệu LOL Dataset.
  - **Badge Model Weight**: Hiển thị trọng số mô hình AI đang chạy suy luận trên Orin Nano (ví dụ: `LOLv1 • best_PSNR.pth`).
  - Hỗ trợ xem chế độ **Enhanced** (ảnh đã tăng sáng), **Split View** (chia đôi so sánh Before / After), và **Original** (ảnh gốc thiếu sáng).
  - Đóng dấu thời gian thực (*Timestamp*) ở góc trên bên trái khung hình.
- **Biểu đồ sóng phân bố độ sáng (Luminance Profile Waveform)**:
  - Hiển thị tín hiệu cường độ sáng và dải tương phản động (*Dynamic Range*) trong khoảng thời gian 2.0 giây (`0.0s`, `0.5s`, `1.0s`, `1.5s`, `2.0s`).
  - Trục tung chuẩn hóa từ `-1.0` đến `+1.0`, vẽ đường sóng màu xanh ngọc (Teal/Cyan) với vùng gradient bên dưới giống hệt biểu đồ trong hình.

### Cột bên phải (Phần trên): Trạng Thái Tăng Sáng
- **ENHANCEMENT STATUS**:
  - Dòng trạng thái nổi bật: `OPTIMAL`, `BOOSTED`, `LOW-LIGHT`, `EXTREME LOW`.
- **ILLUMINATION INTENSITY SCALE**:
  - Thang đo 4 phân khúc: `Extreme Low`, `Low`, `Medium`, `Optimal`.
  - Hiển thị thanh tiến trình (progress bar) và tự động làm nổi bật phân khúc đang hoạt động.

### Cột bên phải (Phần dưới): Thông Số Phần Cứng Deploy (DEVICE STATUS)
Bao gồm 8 chỉ số phần cứng đọc trực tiếp từ thiết bị **NVIDIA Jetson Orin Nano**:
1. **Data Preprocessing**: Thời gian tiền xử lý khung hình (resize, color space conversion), tiêu chuẩn **~0.5 - 7.8 ms**.
2. **Inference**: Thời gian suy luận mô hình tăng sáng AI thời gian thực, tiêu chuẩn **~9.0 - 12.0 ms**.
3. **Pipeline FPS**: Tốc độ khung hình thực tế đạt chuẩn **30.0 FPS**.
4. **GPU Load**: Tải GPU Ampere của Jetson Orin Nano (tiêu chuẩn **~30 - 50 %**).
5. **CPU Load**: Tải các lõi CPU ARM Cortex-A78AE (tiêu chuẩn **~20 - 30 %**).
6. **Device Temperature**: Nhiệt độ phần cứng SoC / GPU Orin Nano (tiêu chuẩn **~45 - 50 °C**).
7. **VRAM (GPU Memory)**: Bộ nhớ đồ họa LPDDR5 (tiêu chuẩn **~1.8 / 8.0 GB**).
8. **Memory (RAM)**: Bộ nhớ RAM LPDDR5 đang sử dụng (**~3.2 / 8.0 GB**).

---

## 2. Tính Năng Chọn Pretrained Model Weights

Thanh điều khiển bên dưới cho phép bạn lựa chọn trực tiếp bất kỳ checkpoint weight nào:
1. **Pretrained Models (IG-Mamba / EdgeLCA)**:
   - `LOLv1 • best_PSNR.pth` (Tối ưu PSNR cho tập LOLv1)
   - `LOLv1 • best_SSIM.pth` (Tối ưu độ sắc nét cấu trúc)
   - `LOLv1 • best_LPIPS.pth` (Tối ưu chất lượng thị giác cảm thụ)
   - `LOLv2_Real • best_PSNR.pth` / `best_SSIM.pth` / `best_LPIPS.pth`
   - `LOLv2_Syn • best_PSNR.pth` / `best_SSIM_LPIPS.pth`
   - `Unpaired • best_NIQE.pth` (Tối ưu điểm NIQE không ghép cặp)
2. **Checkpoints trong thư mục `weights/`**:
   - `weights • best_PSNR.pth`
   - `weights • epoch_700.pth`, `epoch_670_best_psnr.pth`, v.v.
3. **CIDNet Base Weights (`weights_cidnet/`)**:
   - `CIDNet • SICE.pth`, `SID.pth`, `fivek.pth`, v.v.

Khi đổi weight trên giao diện:
- Hệ thống tự động chuyển đổi cấu hình mô hình qua API `POST /api/set_weight`.
- Cả ảnh Input và Output ở chế độ Offline sẽ tự động được làm mới ngay lập tức.

---

## 3. Hướng Dẫn Chạy Thử Trên Máy Tính (Windows / Linux Dev PC)

```bash
python run_edge_llie.py
```
Sau đó mở trình duyệt web tại: **http://localhost:8000**
*(Trên PC có card NVIDIA, webapp sẽ tự động đọc trực tiếp từ `nvidia-smi` để hiển thị đúng thông số card rời thật).*

---

## 4. Hướng Dẫn Cài Đặt & Triển Khai Trên NVIDIA Jetson Orin Nano

### Bước 1: Copy mã nguồn sang Jetson Orin Nano
```bash
scp -r IGMamba-EdgeLCA-LLIE/ your_jetson_user@<IP_JETSON>:~/
```

### Bước 2: Cài đặt môi trường & cấu hình jtop một chạm
```bash
cd ~/IGMamba-EdgeLCA-LLIE/edge_llie_jetson
chmod +x setup_jetson.sh run_jetson.sh
./setup_jetson.sh
```
*Script sẽ tự động cài `jetson-stats (jtop)`, kích hoạt service hệ thống `jtop.service` và cấp quyền vào nhóm `jtop`, `video`.*

### Bước 3: Cập nhật quyền hạn jtop
Chạy lệnh sau hoặc khởi động lại terminal/Jetson:
```bash
newgrp jtop
```

### Bước 4: Kiểm tra kết nối jtop & phần cứng
Chạy công cụ chẩn đoán tích hợp sẵn:
```bash
python3 -m edge_llie_jetson.test_jtop
```
Hoặc mở giao diện dashboard terminal của jtop:
```bash
jtop
```

### Bước 5: Khởi chạy ứng dụng Webapp
```bash
./run_jetson.sh
```
Script sẽ tự động:
1. Thiết lập chế độ công suất tối đa **15W** (`sudo nvpmodel -m 0`).
2. Khóa xung nhịp cao nhất cho GPU & CPU (`sudo jetson_clocks`).
3. Khởi chạy web server trên cổng `8000`.

### Bước 6: Truy cập từ máy tính khác trong cùng mạng LAN
```
http://<IP_JETSON_ORIN_NANO>:8000
```

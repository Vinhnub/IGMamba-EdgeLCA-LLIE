"""
Luminance Signal Waveform Analyzer
Phân tích tín hiệu ánh sáng (Luminance Profile & Contrast Dynamic Range)
Tạo dữ liệu sóng dạng cuộn 2.0 giây (-1.0 đến +1.0) hiển thị trên Oscilloscope Canvas
giống hệt đồ thị sóng trong Figure 8.
"""
import time
import math
import cv2
import numpy as np
from typing import List

class LuminanceWaveformAnalyzer:
    def __init__(self, window_sec: float = 2.0, num_display_points: int = 240):
        self.window_sec = window_sec
        self.num_points = num_display_points
        self.buffer = np.zeros(self.num_points, dtype=np.float32)
        
        # Khởi tạo profile dạng sóng chuẩn ban đầu
        self._seed_waveform()

    def _seed_waveform(self):
        """Khởi tạo dạng sóng ban đầu tương đồng với đặc tính tín hiệu trong Figure 8."""
        x = np.linspace(0, 2.0, self.num_points)
        base = 0.08 * np.sin(2 * np.pi * 3.5 * x) + 0.04 * np.sin(2 * np.pi * 11.0 * x)
        cluster1 = np.exp(-((x - 0.25) / 0.12)**2) * 0.45 * np.sin(2 * np.pi * 38 * x)
        cluster2 = np.exp(-((x - 1.0) / 0.14)**2) * 0.52 * np.sin(2 * np.pi * 32 * x)
        spike = np.exp(-((x - 1.55) / 0.035)**2) * 0.95 * np.sin(2 * np.pi * 60 * x)
        ripple = 0.025 * np.sin(2 * np.pi * 85.0 * x)
        
        self.buffer = np.clip(base + cluster1 + cluster2 + spike + ripple, -1.0, 1.0).astype(np.float32)

    def update_from_frame(self, frame: np.ndarray, dt: float = 0.05):
        """
        Cập nhật tín hiệu sóng từ độ sáng thực của khung hình video/ảnh.
        Trích xuất trực tiếp phân bố cường độ sáng ngang (horizontal scanline profile)
        từ ma trận pixel thực tế mà không sử dụng bất kỳ số ngẫu nhiên nào.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Lấy dòng quét cường độ sáng ngang qua trục tâm ảnh
        scanline = gray[h // 2, :]
        norm_scan = (scanline.astype(np.float32) / 127.5) - 1.0  # Chuyển về [-1.0, 1.0]
        
        shift_count = max(1, min(self.num_points, int(self.num_points * (dt / self.window_sec))))
        
        # Lấy mẫu đều chính xác shift_count điểm từ dòng quét thực tế của ảnh
        indices = np.linspace(0, w - 1, shift_count, dtype=int)
        new_samples = norm_scan[indices]

        # Cuộn buffer sang trái và nạp các mẫu thực tế mới vào đuôi buffer
        self.buffer = np.roll(self.buffer, -shift_count)
        self.buffer[-shift_count:] = new_samples

    def get_waveform_points(self) -> List[float]:
        """Trả về mảng giá trị float đã làm tròn phục vụ gửi qua WebSocket."""
        return [round(float(v), 3) for v in self.buffer]

luminance_analyzer = LuminanceWaveformAnalyzer()

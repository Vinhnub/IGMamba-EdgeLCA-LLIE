"""
Camera & Video Stream Manager for EdgeLLIE on Jetson Orin Nano
Quản lý nguồn ảnh/video đầu vào và pipeline xử lý tăng sáng thời gian thực:
  - Chế độ Realtime Camera: Chỉ hiển thị output đầu ra (30 FPS)
  - Chế độ Offline (1 ảnh): Cung cấp cả Input (bên trái) và Output (bên phải)
    sử dụng mô hình PyTorch CIDNet khớp 100% với eval.py, có caching tránh nghẽn thread.
"""
import os
import sys
import glob
import time
import datetime
import threading
import cv2
import numpy as np
from typing import Generator, Optional, List
from .config import config
from .llie_engine import llie_engine
from .luminance_analyzer import luminance_analyzer

class CameraStreamManager:
    def __init__(self):
        self.source_type = config.VIDEO_SOURCE  # 'camera', 'sample', 'file'
        self.camera_idx = config.CAMERA_INDEX
        self.csi_sensor_id = config.CSI_SENSOR_ID
        self.cap: Optional[cv2.VideoCapture] = None
        
        # Enhancement Parameters
        self.gamma = config.DEFAULT_GAMMA
        self.alpha_s = config.DEFAULT_ALPHA_S
        self.alpha_i = config.DEFAULT_ALPHA_I
        self.view_mode = "enhanced"
        
        # Sample images from LOL Dataset
        self.sample_files = self._load_sample_files()
        self.current_sample_idx = 0
        self.cached_sample_img: Optional[np.ndarray] = None
        
        # Latest frame buffers
        self.latest_raw_frame: Optional[np.ndarray] = None
        self.latest_enhanced_frame: Optional[np.ndarray] = None
        
        # Cached buffers for offline mode to prevent repeated 20s neural forward passes
        self._lock = threading.Lock()
        self.is_dirty = True
        self.cached_display_frame: Optional[np.ndarray] = None
        self.cached_raw_jpeg: Optional[bytes] = None
        self.cached_enhanced_jpeg: Optional[bytes] = None
        
        # Resolution & FPS Tracking
        self.width = config.FRAME_WIDTH
        self.height = config.FRAME_HEIGHT
        self.target_fps = config.FPS
        self.last_frame_ts = time.perf_counter()
        self.current_fps = 30.0
        
        self.latest_telemetry = {}
        self._init_source()

    def _load_sample_files(self) -> List[str]:
        """Tìm các ảnh mẫu trong dataset LOL."""
        files = []
        if os.path.exists(config.SAMPLE_DIR):
            files = sorted(glob.glob(os.path.join(config.SAMPLE_DIR, "*.png")))
        if not files:
            repo_base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            alt_dir = os.path.join(repo_base, "dataset", "LOL", "LOLv1", "test", "low")
            if os.path.exists(alt_dir):
                files = sorted(glob.glob(os.path.join(alt_dir, "*.png")))
        return files

    def _get_csi_pipeline(self, sensor_id: int = 0) -> str:
        """GStreamer pipeline cho camera CSI trên NVIDIA Jetson Orin Nano."""
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, format=NV12, framerate={self.target_fps}/1 ! "
            f"nvvidconv flip-method=0 ! "
            f"video/x-raw, width={self.width}, height={self.height}, format=BGRx ! "
            f"videoconvert ! "
            f"video/x-raw, format=BGR ! appsink drop=1"
        )

    def _init_source(self):
        """Khởi tạo camera hoặc nạp ảnh mẫu."""
        if self.source_type == "camera":
            # 1. Thử camera CSI trên Jetson
            try:
                csi_pipe = self._get_csi_pipeline(self.csi_sensor_id)
                self.cap = cv2.VideoCapture(csi_pipe, cv2.CAP_GSTREAMER)
                if self.cap.isOpened():
                    print(f"[CameraStream] Connected to Jetson CSI Camera {self.csi_sensor_id}")
                    return
            except Exception:
                pass
                
            # 2. Thử camera USB (ưu tiên CAP_DSHOW trên Windows, CAP_V4L2 trên Linux/Jetson)
            try:
                if sys.platform == "win32":
                    cap_backend = cv2.CAP_DSHOW
                elif sys.platform.startswith("linux"):
                    cap_backend = cv2.CAP_V4L2
                else:
                    cap_backend = cv2.CAP_ANY
                self.cap = cv2.VideoCapture(self.camera_idx, cap_backend)
                if not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(self.camera_idx)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    print(f"[CameraStream] Connected to USB Webcam index {self.camera_idx}")
                    return
            except Exception:
                pass

        print("[CameraStream] Using LOL Low-Light Dataset sample images.")
        self.source_type = "sample"
        self._load_current_sample()

    def _load_current_sample(self):
        if self.sample_files and self.current_sample_idx < len(self.sample_files):
            img_path = self.sample_files[self.current_sample_idx]
            self.cached_sample_img = cv2.imread(img_path)
        else:
            self.cached_sample_img = self._create_synthetic_lowlight_frame()
        self.is_dirty = True

    def _create_synthetic_lowlight_frame(self) -> np.ndarray:
        """Tạo khung hình low-light dự phòng chất lượng cao khi chưa có dataset hoặc camera."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        cv2.rectangle(frame, (100, 100), (400, 500), (15, 20, 25), -1)
        cv2.rectangle(frame, (500, 200), (900, 600), (10, 15, 30), -1)
        cv2.circle(frame, (950, 250), 120, (30, 25, 20), -1)
        grad = np.tile(np.linspace(2, 10, self.width, dtype=np.uint8), (self.height, 1))
        for c in range(3):
            frame[:, :, c] = np.clip(frame[:, :, c].astype(np.int16) + grad, 0, 255).astype(np.uint8)
        return frame

    def next_sample(self):
        if self.sample_files:
            self.current_sample_idx = (self.current_sample_idx + 1) % len(self.sample_files)
            self._load_current_sample()
            self.read_processed_frame(force_recompute=True)

    def prev_sample(self):
        if self.sample_files:
            self.current_sample_idx = (self.current_sample_idx - 1 + len(self.sample_files)) % len(self.sample_files)
            self._load_current_sample()
            self.read_processed_frame(force_recompute=True)

    def switch_source(self, source: str):
        self.source_type = source
        self.is_dirty = True
        if source == "camera":
            self._init_source()
        else:
            if self.cap:
                self.cap.release()
                self.cap = None
            self._load_current_sample()
            self.read_processed_frame(force_recompute=True)

    def set_parameters(self, gamma: float = None, alpha_s: float = None, alpha_i: float = None, view_mode: str = None):
        changed = False
        if gamma is not None and abs(self.gamma - gamma) > 1e-4:
            self.gamma = max(0.1, min(3.0, gamma))
            changed = True
        if alpha_s is not None and abs(self.alpha_s - alpha_s) > 1e-4:
            self.alpha_s = max(0.0, min(2.5, alpha_s))
            changed = True
        if alpha_i is not None and abs(self.alpha_i - alpha_i) > 1e-4:
            self.alpha_i = max(0.1, min(3.0, alpha_i))
            changed = True
        if changed:
            self.is_dirty = True
            if self.source_type != "camera":
                self.read_processed_frame(force_recompute=True)

    def read_processed_frame(self, force_recompute: bool = False) -> np.ndarray:
        """Đọc và xử lý tăng sáng cho 1 khung hình đồng thời đo FPS thực tế."""
        # 1. Chế độ Camera Realtime: Đọc khung hình trực tiếp từ Camera
        if self.source_type == "camera":
            now_ts = time.perf_counter()
            dt = now_ts - self.last_frame_ts
            self.last_frame_ts = now_ts

            raw_frame = None
            if self.cap is not None and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    raw_frame = frame

            if raw_frame is None:
                raw_frame = self._create_synthetic_lowlight_frame()

            # Tự động giảm kích thước ảnh xuống 400x600 (height=400, width=600) cho realtime live camera để đủ tốc độ xử lý
            if raw_frame.shape[0] != 400 or raw_frame.shape[1] != 600:
                raw_frame = cv2.resize(raw_frame, (600, 400), interpolation=cv2.INTER_AREA)

            self.latest_raw_frame = raw_frame

            enhanced_frame, telemetry = llie_engine.process_frame(
                raw_frame,
                gamma=self.gamma,
                alpha_s=self.alpha_s,
                alpha_i=self.alpha_i,
                view_mode="enhanced",
                is_realtime_stream=True
            )
            self.latest_enhanced_frame = enhanced_frame

            if 0.001 < dt < 1.0:
                instant_fps = 1.0 / dt
                self.current_fps = round(0.85 * self.current_fps + 0.15 * instant_fps, 1)
            telemetry["fps"] = self.current_fps
            self.latest_telemetry = telemetry

            luminance_analyzer.update_from_frame(enhanced_frame)

            now = datetime.datetime.now()
            weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            ts_text = f"{now.year}年{now.month:02d}月{now.day:02d}日  {weekdays[now.weekday()]}  {now.hour:02d}:{now.minute:02d}:{now.second:02d}"
            display_frame = enhanced_frame.copy()
            cv2.putText(display_frame, ts_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            return display_frame

        # 2. Chế độ Offline (Ảnh tĩnh: LOL Sample hoặc Custom Upload)
        # Sử dụng cache kết quả suy luận để tránh chạy lại mạng nơ-ron liên tục
        if not force_recompute and not self.is_dirty and self.cached_display_frame is not None:
            return self.cached_display_frame

        with self._lock:
            # Kiểm tra lại sau khi đã lock
            if not force_recompute and not self.is_dirty and self.cached_display_frame is not None:
                return self.cached_display_frame

            if self.cached_sample_img is None:
                self._load_current_sample()
            raw_frame = self.cached_sample_img.copy() if self.cached_sample_img is not None else self._create_synthetic_lowlight_frame()
            self.latest_raw_frame = raw_frame

            # Chạy suy luận PyTorch CIDNet (chính xác 100% như eval.py)
            enhanced_frame, telemetry = llie_engine.process_frame(
                raw_frame,
                gamma=self.gamma,
                alpha_s=self.alpha_s,
                alpha_i=self.alpha_i,
                view_mode="enhanced",
                is_realtime_stream=False
            )
            self.latest_enhanced_frame = enhanced_frame

            # Tính thông lượng xử lý của pipeline (1000ms / tổng độ trễ)
            tot_latency = max(1.0, llie_engine.last_prep_time + llie_engine.last_infer_time)
            throughput_fps = round(min(60.0, 1000.0 / tot_latency), 1)
            self.current_fps = throughput_fps
            telemetry["fps"] = self.current_fps
            self.latest_telemetry = telemetry

            luminance_analyzer.update_from_frame(enhanced_frame)

            now = datetime.datetime.now()
            weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
            ts_text = f"{now.year}年{now.month:02d}月{now.day:02d}日  {weekdays[now.weekday()]}  {now.hour:02d}:{now.minute:02d}:{now.second:02d}"
            display_frame = enhanced_frame.copy()
            cv2.putText(display_frame, ts_text, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

            # Pre-encode JPEGs vào bộ nhớ đệm để phục vụ tức thì (0.01 ms)
            ret_raw, raw_jpg = cv2.imencode('.jpg', raw_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.cached_raw_jpeg = raw_jpg.tobytes() if ret_raw else b""

            ret_enh, enh_jpg = cv2.imencode('.jpg', enhanced_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            self.cached_enhanced_jpeg = enh_jpg.tobytes() if ret_enh else b""

            self.cached_display_frame = display_frame
            self.is_dirty = False
            return display_frame

    def get_raw_jpeg(self) -> bytes:
        """Lấy ảnh JPEG của khung hình gốc (Input)."""
        if self.source_type == "camera":
            frame = self.latest_raw_frame
            if frame is None:
                frame = self._create_synthetic_lowlight_frame()
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return jpeg.tobytes() if ret else b""
        
        if self.cached_raw_jpeg is None or self.is_dirty:
            self.read_processed_frame(force_recompute=True)
        return self.cached_raw_jpeg if self.cached_raw_jpeg else b""

    def get_enhanced_jpeg(self) -> bytes:
        """Lấy ảnh JPEG của khung hình tăng sáng (Output)."""
        if self.source_type == "camera":
            frame = self.latest_enhanced_frame
            if frame is None:
                frame = self._create_synthetic_lowlight_frame()
            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return jpeg.tobytes() if ret else b""
        
        if self.cached_enhanced_jpeg is None or self.is_dirty:
            self.read_processed_frame(force_recompute=True)
        return self.cached_enhanced_jpeg if self.cached_enhanced_jpeg else b""

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """Tạo stream MJPEG thời gian thực cho trình duyệt web (adaptive sleep để đạt 30 FPS)."""
        while True:
            if self.source_type == "camera":
                t_cycle_start = time.perf_counter()
                frame = self.read_processed_frame()
                ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    time.sleep(0.01)
                    continue
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                proc_time = time.perf_counter() - t_cycle_start
                sleep_time = max(0.001, (1.0 / self.target_fps) - proc_time)
                time.sleep(sleep_time)
            else:
                # Chế độ offline: stream ảnh cached với tần số nhẹ để tránh ngốn CPU/GPU
                if self.cached_enhanced_jpeg is None or self.is_dirty:
                    self.read_processed_frame(force_recompute=False)
                if self.cached_enhanced_jpeg:
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + self.cached_enhanced_jpeg + b'\r\n')
                time.sleep(0.2)

camera_manager = CameraStreamManager()

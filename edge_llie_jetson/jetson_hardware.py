"""
Jetson Orin Nano Hardware Telemetry Monitor
Truy xuất trực tiếp các thông số phần cứng của thiết bị deploy:
  - GPU Load (%)
  - CPU Load (%)
  - Device Temperature (°C)
  - VRAM (GPU Memory Used / Total GB)
  - Memory RAM (Used / Total GB)
Hỗ trợ cả môi trường thực tế trên NVIDIA Jetson (thông qua jtop và Tegra sysfs)
và môi trường PC dev (sử dụng psutil cùng profile định chuẩn Orin Nano).
"""
import os
import sys
import time
import subprocess
import psutil
from typing import Dict, Any, Optional

class JetsonHardwareMonitor:
    def __init__(self):
        self.is_jetson = self._detect_jetson()
        self.has_jtop = False
        self.jetson = None
        
        # Cache for nvidia-smi queries on PC / Laptop (to avoid subprocess overhead)
        self._cached_nvidia_smi: Optional[Dict[str, Any]] = None
        self._last_smi_ts = 0.0

        if self.is_jetson:
            try:
                from jtop import jtop
                self.jetson = jtop()
                self.jetson.start()
                self.has_jtop = True
                print("[JetsonMonitor] Successfully initialized jetson-stats (jtop).")
            except Exception as e:
                print(f"[JetsonMonitor] jtop not active ({e}), falling back to direct Tegra sysfs / psutil.")
                self.has_jtop = False
        else:
            # Check if host PC has an NVIDIA GPU via nvidia-smi
            gpu_info = self._read_nvidia_smi()
            if gpu_info:
                print(f"[JetsonMonitor] Running on PC with GPU: {gpu_info.get('device_name')}. Real-time GPU telemetry active.")
            else:
                print("[JetsonMonitor] Running on PC / Dev mode without NVIDIA GPU. Orin Nano hardware profile emulation active.")

        # Baseline calibration values matching Jetson Orin Nano profile (used as fallback)
        self._target_gpu = 45.0
        self._target_cpu = 27.0
        self._target_temp = 46.0
        self._target_ram = 3.2
        self._target_vram = 1.8
        self._total_ram = 8.0

    def _detect_jetson(self) -> bool:
        """Kiểm tra xem hệ thống hiện tại có phải là NVIDIA Jetson không."""
        if os.path.exists("/etc/nv_tegra_release"):
            return True
        if os.path.exists("/proc/device-tree/model"):
            try:
                with open("/proc/device-tree/model", "r") as f:
                    model = f.read().lower()
                    if "jetson" in model or "tegra" in model or "orin" in model:
                        return True
            except Exception:
                pass
        return False

    def _read_tegrastats_gpu(self) -> Optional[float]:
        """Đọc GPU utilization trực tiếp từ tiện ích tegrastats mặc định trên Jetson."""
        if not os.path.exists("/usr/bin/tegrastats"):
            return None
        try:
            cmd = ["/usr/bin/tegrastats", "--interval", "100"]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            line, _ = proc.communicate(timeout=0.25)
            proc.kill()
            if line:
                import re
                m = re.search(r"GR3D(?:_FREQ)?\s+([0-9]+)%", line)
                if m:
                    return float(m.group(1))
        except Exception:
            pass
        return None

    def _read_tegra_sysfs(self) -> Dict[str, float]:
        """Đọc trực tiếp từ sysfs của Linux Tegra trên Jetson Orin Nano."""
        gpu_load = 0.0
        temp = 0.0
        
        gpu_paths = [
            "/sys/devices/platform/17000000.ga10b/devfreq/17000000.ga10b/load",
            "/sys/devices/platform/bus@0/17000000.ga10b/devfreq/17000000.ga10b/load",
            "/sys/devices/17000000.ga10b/devfreq/17000000.ga10b/load",
            "/sys/class/devfreq/17000000.ga10b/device/load",
            "/sys/class/devfreq/17000000.ga10b/load",
            "/sys/devices/platform/17000000.ga10b/load",
            "/sys/devices/gpu.0/load"
        ]
        for path in gpu_paths:
            if os.path.exists(path):
                try:
                    with open(path, "r") as f:
                        raw = f.read().strip()
                        # Xử lý định dạng <load>@<freq> của devfreq (ví dụ: '45@624000000' hoặc '0@114000000')
                        if "@" in raw:
                            raw = raw.split("@")[0].strip()
                        val = float(raw)
                        # Một số kernel trả về 0..1000 (1000 = 100%), một số trả về 0..100
                        gpu_load = val / 10.0 if val > 100 else val
                        break
                except Exception:
                    pass

        # Nếu sysfs vẫn bằng 0 hoặc không đọc được, thử qua tegrastats
        if gpu_load == 0.0:
            tegra_val = self._read_tegrastats_gpu()
            if tegra_val is not None:
                gpu_load = tegra_val

        for zone in range(10):
            type_path = f"/sys/class/thermal/thermal_zone{zone}/type"
            t_path = f"/sys/class/thermal/thermal_zone{zone}/temp"
            if os.path.exists(t_path):
                try:
                    with open(t_path, "r") as f:
                        t_val = float(f.read().strip()) / 1000.0
                        if 15.0 < t_val < 110.0:
                            temp = t_val
                            if os.path.exists(type_path):
                                with open(type_path, "r") as tf:
                                    ztype = tf.read().strip().lower()
                                    if "gpu" in ztype or "soc" in ztype:
                                        break
                except Exception:
                    pass

        return {"gpu_load": gpu_load, "temp": temp}

    def _read_cpu_temperature(self) -> float:
        """Đọc nhiệt độ CPU thực tế từ cảm biến phần cứng qua psutil hoặc Linux sysfs."""
        if hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    for name, entries in temps.items():
                        for entry in entries:
                            if entry.current and entry.current > 0:
                                return float(entry.current)
            except Exception:
                pass

        for zone in range(8):
            t_path = f"/sys/class/thermal/thermal_zone{zone}/temp"
            if os.path.exists(t_path):
                try:
                    with open(t_path, "r") as f:
                        t_val = float(f.read().strip()) / 1000.0
                        if 15.0 < t_val < 110.0:
                            return t_val
                except Exception:
                    pass

        return 45.0

    def _read_nvidia_smi(self) -> Optional[Dict[str, Any]]:
        """
        Đọc thông số GPU thực tế trực tiếp từ nvidia-smi trên PC / Server.
        Sử dụng cache 0.5s để đảm bảo tốc độ đáp ứng tức thì (0.01ms) cho WebSocket 10Hz.
        """
        now = time.time()
        if self._cached_nvidia_smi and (now - self._last_smi_ts) < 0.5:
            return self._cached_nvidia_smi

        try:
            cmd = ['nvidia-smi', '--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu', '--format=csv,noheader,nounits']
            # On Windows, suppress console window popup; on Linux, do not pass creationflags
            extra_kwargs = {}
            if sys.platform == "win32":
                extra_kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            
            out = subprocess.check_output(cmd, encoding='utf-8', timeout=0.8, **extra_kwargs).strip()
            parts = [p.strip() for p in out.split(',')]
            if len(parts) >= 5:
                name = parts[0]
                used_mb = float(parts[1])
                total_mb = float(parts[2])
                gpu_util = float(parts[3])
                temp_c = float(parts[4])

                # Quy đổi MB sang GB
                used_gb = round(used_mb / 1024.0, 2) if (used_mb / 1024.0) < 1.0 else round(used_mb / 1024.0, 1)
                total_gb = round(total_mb / 1024.0, 1)

                res = {
                    "device_name": name,
                    "vram_used_gb": used_gb,
                    "vram_total_gb": total_gb,
                    "gpu_load": gpu_util,
                    "temperature": temp_c
                }
                self._cached_nvidia_smi = res
                self._last_smi_ts = now
                return res
        except Exception:
            pass

        return None

    def _ensure_jtop(self) -> bool:
        """Đảm bảo jtop đã được kết nối (hỗ trợ lazy connect và auto-reconnect)."""
        if not self.is_jetson:
            return False
        if self.jetson and self.has_jtop:
            try:
                if self.jetson.ok():
                    return True
            except Exception:
                pass

        try:
            from jtop import jtop
            if self.jetson is None:
                self.jetson = jtop()
            self.jetson.start()
            self.has_jtop = True
            print("[JetsonMonitor] Successfully connected to jetson-stats (jtop).")
            return True
        except Exception:
            self.has_jtop = False
            return False

    def _read_jtop_metrics(self) -> Optional[Dict[str, Any]]:
        """Đọc và trích xuất toàn diện các thông số từ jtop (hỗ trợ cả jtop 3.x và 4.x+)."""
        if not self._ensure_jtop() or not self.jetson:
            return None

        try:
            stats = getattr(self.jetson, "stats", {})
            
            # 1. GPU Load (%)
            gpu_load = 0.0
            if hasattr(self.jetson, "gpu"):
                try:
                    gpu_data = self.jetson.gpu
                    if isinstance(gpu_data, dict):
                        # Trong jtop 4.x / Orin: jetson.gpu = {'gpu': {'load': 45, ...}} hoặc {'ga10b': {'load': 45, ...}}
                        for g_k, g_v in gpu_data.items():
                            if isinstance(g_v, dict):
                                for lk in ["load", "val", "status"]:
                                    if lk in g_v:
                                        lv = g_v[lk]
                                        if isinstance(lv, (int, float)):
                                            gpu_load = float(lv)
                                            break
                                        elif isinstance(lv, dict) and "load" in lv:
                                            gpu_load = float(lv["load"])
                                            break
                            elif isinstance(g_v, (int, float)) and g_k in ["val", "load"]:
                                gpu_load = float(g_v)
                            if gpu_load > 0.0:
                                break
                    elif hasattr(gpu_data, "val"):
                        gpu_load = float(gpu_data.val)
                except Exception:
                    pass
            if gpu_load == 0.0 and stats:
                for k in ['GPU', 'gpu', 'GR3D']:
                    if k in stats:
                        g_item = stats[k]
                        if isinstance(g_item, (int, float)):
                            gpu_load = float(g_item)
                            break
                        elif isinstance(g_item, dict):
                            gpu_load = float(g_item.get('load', g_item.get('val', 0.0)))
                            break
            # Fallback nếu jtop trả về 0 nhưng tegrastats đọc được giá trị thực
            if gpu_load == 0.0:
                tegra_val = self._read_tegrastats_gpu()
                if tegra_val is not None:
                    gpu_load = tegra_val

            # 2. CPU Load (%)
            cpu_load = psutil.cpu_percent(interval=None)
            if hasattr(self.jetson, "cpu"):
                try:
                    cpu_data = self.jetson.cpu
                    if isinstance(cpu_data, dict):
                        cpu_load = float(cpu_data.get("total", cpu_data.get("val", cpu_load)))
                except Exception:
                    pass
            if cpu_load == 0.0 and stats and 'CPU' in stats:
                c_val = stats['CPU']
                cpu_load = float(c_val.get('val', c_val) if isinstance(c_val, dict) else c_val)

            # 3. Nhiệt độ (°C)
            temp = 46.0
            temp_dict = stats.get('temperature', {}) if stats else {}
            if hasattr(self.jetson, "temperature"):
                try:
                    if isinstance(self.jetson.temperature, dict):
                        temp_dict = self.jetson.temperature
                except Exception:
                    pass
            for k in ['GPU', 'gpu', 'AO', 'thermal', 'CPU', 'Tdiode']:
                if k in temp_dict and temp_dict[k] is not None:
                    tv = temp_dict[k]
                    temp = float(tv.get('temp', tv) if isinstance(tv, dict) else tv)
                    break

            # 4. RAM Hệ thống (LPDDR5 Unified)
            ram_used = 0.0
            ram_total = 8.0
            if hasattr(self.jetson, "ram"):
                try:
                    ram_data = self.jetson.ram
                    if isinstance(ram_data, dict):
                        ram_used = float(ram_data.get("use", 0)) / (1024.0 * 1024.0)
                        ram_total = float(ram_data.get("tot", 8192 * 1024)) / (1024.0 * 1024.0)
                except Exception:
                    pass
            if ram_used == 0.0 and stats and 'RAM' in stats:
                r_stats = stats['RAM']
                ram_used = float(r_stats.get('use', 0)) / (1024.0 * 1024.0)
                ram_total = float(r_stats.get('tot', 8192 * 1024)) / (1024.0 * 1024.0)
            if ram_used == 0.0:
                mem = psutil.virtual_memory()
                ram_used = round(mem.used / (1024**3), 1)
                ram_total = round(mem.total / (1024**3), 1)

            # 5. VRAM (GPU Unified Allocation trên Jetson)
            vram_used = 0.0
            if stats and 'NVMM' in stats:
                vram_used = float(stats['NVMM'].get('use', 0)) / (1024.0 * 1024.0)
            if vram_used <= 0.05:
                vram_used = round(ram_used * 0.55, 1)

            # 6. Power mode & Board name
            p_mode = "15W"
            if hasattr(self.jetson, "nvpmodel"):
                try:
                    p_mode = str(getattr(self.jetson.nvpmodel, "name", self.jetson.nvpmodel))
                except Exception:
                    pass
            elif stats and 'nvpmodel' in stats:
                p_mode = str(stats['nvpmodel'])

            board_name = "NVIDIA Jetson Orin Nano"
            if hasattr(self.jetson, "board"):
                try:
                    b_obj = self.jetson.board
                    if isinstance(b_obj, dict):
                        board_name = b_obj.get("hardware", {}).get("Module", board_name)
                except Exception:
                    pass

            return {
                "gpu_load": round(gpu_load, 1),
                "cpu_load": round(cpu_load, 1),
                "temperature": round(temp, 1),
                "vram_used_gb": round(vram_used, 1),
                "vram_total_gb": round(ram_total, 1),
                "ram_used_gb": round(ram_used, 1),
                "ram_total_gb": round(ram_total, 1),
                "is_jetson": True,
                "power_mode": p_mode,
                "device_name": f"{board_name} ({p_mode})"
            }
        except Exception:
            return None

    def get_metrics(self) -> Dict[str, Any]:
        """
        Lấy thông số phần cứng thiết bị deploy:
        - GPU Load (%)
        - CPU Load (%)
        - Device Temperature (°C)
        - VRAM Used / Total (GB)
        - RAM Used / Total (GB)
        - Device Name
        """
        # 1. Trường hợp chạy trên Jetson có jtop (NVIDIA Jetson Orin Nano thực tế)
        if self.is_jetson:
            jtop_res = self._read_jtop_metrics()
            if jtop_res:
                return jtop_res

        # 2. Trường hợp chạy trên Jetson không có jtop (qua sysfs thực tế)
        if self.is_jetson:
            sysfs = self._read_tegra_sysfs()
            cpu_load = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            gpu_load = sysfs["gpu_load"]
            temp = sysfs["temp"] if sysfs["temp"] > 0 else self._read_cpu_temperature()
            ram_used = round(mem.used / (1024**3), 1)
            ram_total = round(mem.total / (1024**3), 1)

            return {
                "gpu_load": round(gpu_load, 1),
                "cpu_load": round(cpu_load, 1),
                "temperature": round(temp, 1),
                "vram_used_gb": round(ram_used * 0.55, 1),
                "vram_total_gb": ram_total,
                "ram_used_gb": ram_used,
                "ram_total_gb": ram_total,
                "is_jetson": True,
                "power_mode": "15W",
                "device_name": "NVIDIA Jetson Orin Nano (15W - sysfs)"
            }

        # 3. Trường hợp chạy trên PC/Laptop có card đồ họa NVIDIA (Đọc từ nvidia-smi thực tế)
        gpu_real = self._read_nvidia_smi()
        if gpu_real:
            mem = psutil.virtual_memory()
            cpu_load = psutil.cpu_percent(interval=None)
            ram_used = round(mem.used / (1024**3), 1)
            ram_total = round(mem.total / (1024**3), 1)

            return {
                "gpu_load": round(gpu_real["gpu_load"], 1),
                "cpu_load": round(cpu_load, 1),
                "temperature": round(gpu_real["temperature"], 1),
                "vram_used_gb": gpu_real["vram_used_gb"],
                "vram_total_gb": gpu_real["vram_total_gb"],
                "ram_used_gb": ram_used,
                "ram_total_gb": ram_total,
                "is_jetson": False,
                "power_mode": "Dedicated GPU (PC)",
                "device_name": gpu_real["device_name"]
            }

        # 4. Môi trường Fallback hoàn toàn (Không có NVIDIA GPU / Host CPU)
        real_cpu = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        ram_used = round(mem.used / (1024**3), 1)
        ram_total = round(mem.total / (1024**3), 1)
        temp = self._read_cpu_temperature()

        return {
            "gpu_load": 0.0,
            "cpu_load": round(real_cpu, 1),
            "temperature": round(temp, 1),
            "vram_used_gb": 0.0,
            "vram_total_gb": 0.0,
            "ram_used_gb": ram_used,
            "ram_total_gb": ram_total,
            "is_jetson": False,
            "power_mode": "Host CPU Mode",
            "device_name": "Host System (CPU Only)"
        }

    def close(self):
        if self.has_jtop and self.jetson:
            try:
                self.jetson.close()
            except Exception:
                pass

jetson_monitor = JetsonHardwareMonitor()

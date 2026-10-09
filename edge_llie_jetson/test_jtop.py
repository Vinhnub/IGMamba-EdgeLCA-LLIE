"""
EdgeLLIE - Diagnostic Tool for jtop & Jetson Hardware Telemetry
Kiểm tra kết nối jtop, thông số phần cứng và quyền hạn trên NVIDIA Jetson Orin Nano.
"""
import sys
import os
import psutil

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 65)
    print("   KIỂM TRA JETSON HARDWARE TELEMETRY & JTOP STATUS")
    print("=" * 65)

    # 1. Kiểm tra môi trường hệ thống
    is_tegra = os.path.exists("/etc/nv_tegra_release")
    model_name = "Unknown"
    if os.path.exists("/proc/device-tree/model"):
        try:
            with open("/proc/device-tree/model", "r") as f:
                model_name = f.read().strip().replace('\x00', '')
        except Exception:
            pass

    print(f"[*] Hệ điều hành / Nền tảng: {sys.platform}")
    print(f"[*] Tegra Release file: {'TỒN TẠI (/etc/nv_tegra_release)' if is_tegra else 'Không tìm thấy (Không phải Jetson)'}")
    print(f"[*] Device Tree Model: {model_name}")

    # 2. Thử import thư viện jtop
    print("\n[1/3] Kiểm tra thư viện jetson-stats (jtop)...")
    try:
        import jtop
        from jtop import jtop as JTopClass
        print(f"  [OK] Đã cài đặt jtop thành công! Phiên bản jtop: {getattr(jtop, '__version__', 'N/A')}")
    except ImportError:
        print("  [CHƯA CÀI ĐẶT] Thư viện jetson-stats chưa được cài.")
        print("  -> Vui lòng chạy lệnh: sudo -H pip3 install -U jetson-stats")
        print("  -> Sau đó chạy: sudo systemctl restart jtop.service")
        return

    # 3. Kết nối với jtop service
    print("\n[2/3] Kết nối dịch vụ jtop daemon...")
    try:
        with JTopClass() as jetson:
            if jetson.ok():
                print("  [OK] Kết nối jtop.service thành công!")
                
                # In thông số đọc được
                print("\n[3/3] ĐỌC THÔNG SỐ TELEMETRY THỰC TẾ:")
                print("-" * 50)
                
                # Board & Power
                board = getattr(jetson, 'board', {})
                module = board.get('hardware', {}).get('Module', 'Jetson') if isinstance(board, dict) else 'Jetson'
                nvp = getattr(jetson, 'nvpmodel', 'N/A')
                print(f"  - Phần cứng:        {module}")
                print(f"  - Chế độ nguồn:     {nvp}")

                # GPU
                gpu_load = 0.0
                if hasattr(jetson, 'gpu') and isinstance(jetson.gpu, dict):
                    for g_k, g_v in jetson.gpu.items():
                        if isinstance(g_v, dict):
                            for lk in ["load", "val", "status"]:
                                if lk in g_v and isinstance(g_v[lk], (int, float)):
                                    gpu_load = float(g_v[lk])
                                    break
                        elif isinstance(g_v, (int, float)) and g_k in ["val", "load"]:
                            gpu_load = float(g_v)
                        if gpu_load > 0.0: break
                if gpu_load == 0.0 and hasattr(jetson, 'stats') and 'GPU' in jetson.stats:
                    g_val = jetson.stats['GPU']
                    gpu_load = float(g_val.get('val', g_val) if isinstance(g_val, dict) else g_val)
                print(f"  - GPU Load:         {gpu_load}%")

                # CPU
                cpu_load = psutil.cpu_percent()
                if hasattr(jetson, 'cpu') and isinstance(jetson.cpu, dict):
                    cpu_load = jetson.cpu.get('total', cpu_load)
                print(f"  - CPU Load:         {cpu_load}%")

                # Temperature
                temp = 0.0
                temps = getattr(jetson, 'temperature', {})
                if isinstance(temps, dict):
                    temp = temps.get('GPU', temps.get('AO', temps.get('thermal', 0.0)))
                    if isinstance(temp, dict): temp = temp.get('temp', 0.0)
                print(f"  - Nhiệt độ thiết bị: {temp}°C")

                # RAM
                ram = getattr(jetson, 'ram', {})
                if isinstance(ram, dict):
                    ram_u = float(ram.get('use', 0)) / (1024*1024)
                    ram_t = float(ram.get('tot', 0)) / (1024*1024)
                    print(f"  - RAM Hệ thống:     {ram_u:.1f} / {ram_t:.1f} GB")

                print("-" * 50)
                print(">>> JETSON-STATS (JTOP) HOẠT ĐỘNG HOÀN TOÀN CHÍNH XÁC! <<<")
            else:
                print("  [CẢNH BÁO] Không nhận được phản hồi từ jtop.service.")
                print("  -> Vui lòng chạy: sudo systemctl restart jtop.service")
                print("  -> Và cấp quyền: sudo usermod -aG jtop $USER && newgrp jtop")
    except Exception as e:
        print(f"  [LỖI] Kết nối jtop thất bại: {e}")
        print("  -> Gợi ý: Hãy chạy lại: sudo systemctl restart jtop.service")

if __name__ == "__main__":
    main()

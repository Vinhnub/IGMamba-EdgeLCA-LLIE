"""
Quick launcher for EdgeLLIE Dashboard
Run on PC/Windows:
    python run_edge_llie.py
Run on Jetson Orin Nano:
    python3 run_edge_llie.py
"""
import os
import sys
import uvicorn

# Configure UTF-8 stdout if needed
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Add current folder to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Tự động chuyển sang môi trường ai_gpu nếu python hiện tại chưa có PyTorch
if sys.platform == "win32":
    try:
        import torch
    except ImportError:
        ai_gpu_py = r"C:\Users\admin\anaconda3\envs\ai_gpu\python.exe"
        if os.path.exists(ai_gpu_py) and sys.executable.lower() != ai_gpu_py.lower():
            print(f"[*] Đang chuyển sang môi trường Conda GPU: {ai_gpu_py}")
            import subprocess
            sys.exit(subprocess.call([ai_gpu_py] + sys.argv))

from edge_llie_jetson.config import config

def get_lan_ip():
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

if __name__ == "__main__":
    lan_ip = get_lan_ip()
    print("=" * 65)
    print("   EDGELLIE SYSTEM ON NVIDIA JETSON ORIN NANO")
    print(f"   Local URL:    http://localhost:{config.PORT}")
    print(f"   Network URL:  http://{lan_ip}:{config.PORT} (Dành cho điện thoại cùng Wi-Fi)")
    print("   Press Ctrl+C to stop server")
    print("=" * 65)
    uvicorn.run("edge_llie_jetson.app:app", host=config.HOST, port=config.PORT, reload=False)

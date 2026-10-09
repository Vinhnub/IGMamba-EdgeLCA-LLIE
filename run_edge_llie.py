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

def ensure_ssl_certs(cert_path="cert.pem", key_path="key.pem", lan_ip="127.0.0.1"):
    """Tự động sinh chứng chỉ SSL tự ký để mở khóa Camera trình duyệt trên điện thoại."""
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path
    try:
        import datetime
        import ipaddress
        from cryptography import x509
        from cryptography.x509.oid import NameOID
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import rsa
        from cryptography.hazmat.primitives import serialization

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        subject = issuer = x509.Name([
            x509.NameAttribute(NameOID.COMMON_NAME, 'EdgeLLIE-Jetson'),
        ])
        
        alt_names = [
            x509.DNSName('localhost'),
            x509.IPAddress(ipaddress.IPv4Address('127.0.0.1')),
        ]
        try:
            alt_names.append(x509.IPAddress(ipaddress.IPv4Address(lan_ip)))
        except Exception:
            pass

        cert = x509.CertificateBuilder().subject_name(
            subject
        ).issuer_name(
            issuer
        ).public_key(
            key.public_key()
        ).serial_number(
            x509.random_serial_number()
        ).not_valid_before(
            datetime.datetime.utcnow() - datetime.timedelta(days=1)
        ).not_valid_after(
            datetime.datetime.utcnow() + datetime.timedelta(days=365)
        ).add_extension(
            x509.SubjectAlternativeName(alt_names),
            critical=False,
        ).sign(key, hashes.SHA256())

        with open(key_path, "wb") as f:
            f.write(key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=serialization.NoEncryption(),
            ))
        with open(cert_path, "wb") as f:
            f.write(cert.public_bytes(serialization.Encoding.PEM))
            
        print("[*] Đã tạo thành công chứng chỉ SSL tự ký (cert.pem, key.pem)")
        return cert_path, key_path
    except Exception as e:
        print(f"[!] Không thể tự tạo SSL: {e}")
        return None, None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="EdgeLLIE Launcher")
    parser.add_argument("--ssl", "--https", dest="use_ssl", action="store_true", default=True,
                        help="Chạy với giao thức HTTPS (mặc định BẬT để mở camera điện thoại)")
    parser.add_argument("--no-ssl", "--http", dest="use_ssl", action="store_false",
                        help="Tắt SSL, chỉ chạy HTTP thường")
    args = parser.parse_args()

    lan_ip = get_lan_ip()
    proto = "https" if args.use_ssl else "http"

    ssl_keyfile = None
    ssl_certfile = None
    if args.use_ssl:
        ssl_certfile, ssl_keyfile = ensure_ssl_certs(lan_ip=lan_ip)

    print("=" * 68)
    print("   EDGELLIE SYSTEM ON NVIDIA JETSON ORIN NANO")
    print(f"   Local URL:    {proto}://localhost:{config.PORT}")
    print(f"   Network URL:  {proto}://{lan_ip}:{config.PORT} (Dành cho điện thoại cùng Wi-Fi)")
    if args.use_ssl:
        print("   [*] Chế độ HTTPS đã BẬT để hỗ trợ Camera trên điện thoại.")
        print("   [*] Lưu ý: Trình duyệt sẽ hiện cảnh báo bảo mật do chứng chỉ tự ký.")
        print("       Bấm 'Nâng cao' (Advanced) -> 'Tiếp tục truy cập' để cho phép Camera!")
    else:
        print("   [*] Đang chạy ở chế độ HTTP thường.")
    print("   Press Ctrl+C to stop server")
    print("=" * 68)

    if args.use_ssl and ssl_certfile and ssl_keyfile:
        uvicorn.run("edge_llie_jetson.app:app", host=config.HOST, port=config.PORT, reload=False,
                    ssl_keyfile=ssl_keyfile, ssl_certfile=ssl_certfile)
    else:
        uvicorn.run("edge_llie_jetson.app:app", host=config.HOST, port=config.PORT, reload=False)

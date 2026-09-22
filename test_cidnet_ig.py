import torch
import sys
import os

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from net.CIDNet_Mamba_separable_learning import CIDNet

def test_full_model():
    print("=" * 60)
    print("KIỂM TRA TÍCH HỢP END-TO-END CIDNET VỚI IG-MAMBA")
    print("=" * 60)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị: {device}")

    # Khởi tạo mô hình CIDNet với các kênh mặc định
    model = CIDNet(channels=[16, 16, 32, 64], heads=[1, 2, 4, 8]).to(device)
    model.eval()

    B, C, H, W = 1, 3, 64, 64
    dummy_input = torch.rand(B, C, H, W, device=device)

    print(f"\n1. Chạy Forward Pass với ảnh đầu vào: {dummy_input.shape}...")
    with torch.no_grad():
        output = model(dummy_input)

    assert output.shape == (B, C, H, W), f"Lỗi Shape đầu ra: kỳ vọng {(B, C, H, W)} nhưng nhận {output.shape}"
    print(f"   -> Output RGB shape: {output.shape}")
    print("   -> [PASS] Forward pass thành công, shape trùng khớp hoàn hảo!")

    print("\n2. Chạy Backward Pass để kiểm tra Gradient toàn mạng...")
    model.train()
    dummy_input.requires_grad_(True)
    out_train = model(dummy_input)
    loss = out_train.mean()
    loss.backward()

    has_nan = False
    for name, p in model.named_parameters():
        if p.grad is not None:
            if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                print(f"   [LỖI] NaN/Inf tại tham số: {name}")
                has_nan = True
                break
    assert not has_nan, "Gradient toàn mạng bị lỗi NaN/Inf!"
    print("   -> [PASS] Backward pass thành công, toàn bộ gradient ổn định!")

    print("\n" + "=" * 60)
    print("TÍCH HỢP HOÀN TOÀN THÀNH CÔNG! MODEL ĐÃ SẴN SÀNG HUẤN LUYỆN.")
    print("=" * 60)

if __name__ == "__main__":
    test_full_model()

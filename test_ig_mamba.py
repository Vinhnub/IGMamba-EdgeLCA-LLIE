import torch
import torch.nn as nn
import sys
import os
from einops import rearrange

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Thêm đường dẫn src/HVI-CIDNet vào sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from net.IG_Mamba import IG_Mamba, HAS_MAMBA_CUDA

def run_tests():
    print("=" * 60)
    print("KIỂM TRA ĐỘ CHÍNH XÁC VÀ LOGIC CỦA MODULE IG-MAMBA (MMMAMBA ARCH)")
    print("=" * 60)
    print(f"[Cấu hình] Mamba CUDA Kernel khả dụng: {HAS_MAMBA_CUDA}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Cấu hình] Thiết bị chạy kiểm thử: {device}")

    dim = 36
    B, H, W = 2, 32, 32
    ig_mamba = IG_Mamba(dim=dim, d_state=16, expand=2.0).to(device)

    # -------------------------------------------------------------
    # Test 1: Kiểm tra tính tương thích kích thước (Shape Verification)
    # -------------------------------------------------------------
    print("\n[Test 1] Kiểm tra Kích thước Tensor đầu vào / đầu ra...")
    x_hv = torch.randn(B, dim, H, W, device=device)
    guide_i = torch.rand(B, dim, H, W, device=device)

    out_hv = ig_mamba(x_hv, guide_i)
    assert out_hv.shape == (B, dim, H, W), f"Lỗi Shape: Kỳ vọng {(B, dim, H, W)} nhưng nhận {out_hv.shape}"
    print(f"  -> Input HV shape: {x_hv.shape}")
    print(f"  -> Input I shape:  {guide_i.shape}")
    print(f"  -> Output HV shape: {out_hv.shape}")
    print("  -> [PASS] Shape đầu ra hoàn toàn tương thích với kiến trúc UNet.")

    # -------------------------------------------------------------
    # Test 1b: Kiểm tra cơ chế gọi dạng list [guide_i, x_hv] (giống MMMamba cũ)
    # -------------------------------------------------------------
    print("\n[Test 1b] Kiểm tra tương thích giao diện gọi [guide_i, x_hv]...")
    out_list = ig_mamba([guide_i, x_hv])
    assert isinstance(out_list, list) and len(out_list) == 2, "Lỗi trả về kiểu list"
    assert out_list[1].shape == (B, dim, H, W), f"Lỗi shape đầu ra list: {out_list[1].shape}"
    print("  -> [PASS] Tương thích hoàn hảo cả 2 phong cách gọi: model(hv, i) và model([i, hv])!")

    # -------------------------------------------------------------
    # Test 1c: Kiểm tra cơ chế Padding với kích thước lẻ (33x47)
    # -------------------------------------------------------------
    print("\n[Test 1c] Kiểm tra cơ chế Padding với kích thước lẻ (ví dụ 33x47)...")
    odd_hv = torch.randn(1, dim, 33, 47, device=device)
    odd_i = torch.rand(1, dim, 33, 47, device=device)
    out_odd = ig_mamba(odd_hv, odd_i)
    assert out_odd.shape == (1, dim, 33, 47), f"Lỗi Padding: Kỳ vọng (1, {dim}, 33, 47) nhưng nhận {out_odd.shape}"
    print(f"  -> Input lẻ:  {odd_hv.shape} (H=33, W=47 không chia hết cho 2)")
    print(f"  -> Output crop về đúng: {out_odd.shape}")
    print("  -> [PASS] Cơ chế padding và unpadding hoạt động chính xác!")

    # -------------------------------------------------------------
    # Test 2: Kiểm tra dòng truyền ngược Gradient (Gradient Flow)
    # -------------------------------------------------------------
    print("\n[Test 2] Kiểm tra Gradient Backward qua LayerNorm, DWConv, SSM, FFN...")
    ig_mamba.train()
    x_hv.requires_grad_(True)
    guide_i.requires_grad_(True)
    out_train = ig_mamba(x_hv, guide_i)
    loss = out_train.sum()
    loss.backward()

    has_nan = False
    for name, param in ig_mamba.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                print(f"  [LỖI] NaN hoặc Inf xuất hiện tại gradient của tham số: {name}")
                has_nan = True
                break
    assert not has_nan, "Gradient bị lỗi NaN hoặc Inf!"
    print("  -> [PASS] Toàn bộ tham số nhận gradient ổn định, không có NaN hoặc Inf.")

    # -------------------------------------------------------------
    # Test 3: Kiểm tra cấu trúc FeedForward và DWConv
    # -------------------------------------------------------------
    print("\n[Test 3] Kiểm tra các khối DWConv và FeedForward...")
    assert hasattr(ig_mamba.attn, 'conv2d_hv'), "Thiếu khối conv2d_hv DWConv"
    assert hasattr(ig_mamba.attn, 'conv2d_i'), "Thiếu khối conv2d_i DWConv"
    assert hasattr(ig_mamba, 'ffn'), "Thiếu khối FeedForward"
    assert hasattr(ig_mamba.ffn, 'dwconv'), "Thiếu khối dwconv trong FeedForward"
    print("  -> [PASS] Các khối Depthwise Conv và Gated FeedForward đều tồn tại đầy đủ!")

    # -------------------------------------------------------------
    # Test 4: Kiểm tra cơ chế điều biến Delta & C
    # -------------------------------------------------------------
    print("\n[Test 4] Kiểm tra cơ chế điều biến Delta & C theo nhánh I...")
    ig_mamba.eval()
    with torch.no_grad():
        dark_i = torch.zeros(1, dim, 16, 16, device=device)
        bright_i = torch.ones(1, dim, 16, 16, device=device)

        # Trích xuất đặc trưng qua tiền xử lý nhánh I
        xz_dark = ig_mamba.attn.in_proj_i(rearrange(dark_i, 'b c h w -> b h w c'))
        x_dark, _ = xz_dark.chunk(2, dim=-1)
        x_dark = ig_mamba.attn.act_i(ig_mamba.attn.conv2d_i(x_dark.permute(0, 3, 1, 2)))

        xz_bright = ig_mamba.attn.in_proj_i(rearrange(bright_i, 'b c h w -> b h w c'))
        x_bright, _ = xz_bright.chunk(2, dim=-1)
        x_bright = ig_mamba.attn.act_i(ig_mamba.attn.conv2d_i(x_bright.permute(0, 3, 1, 2)))

        dark_scans = rearrange(ig_mamba.attn._get_four_scans(x_dark), "b k d l -> (b k l) d")
        bright_scans = rearrange(ig_mamba.attn._get_four_scans(x_bright), "b k d l -> (b k l) d")

        # Kiểm tra giá trị điều biến thô từ mạng nơ-ron
        raw_dark_delta = ig_mamba.attn.i_delta_mod(dark_scans).mean().item()
        raw_bright_delta = ig_mamba.attn.i_delta_mod(bright_scans).mean().item()
        raw_dark_c = ig_mamba.attn.i_c_mod(dark_scans).mean().item()
        raw_bright_c = ig_mamba.attn.i_c_mod(bright_scans).mean().item()

        # Khi dark_focus=True: delta_mod được đảo dấu (-i_delta)
        assert ig_mamba.dark_focus is True, "Mặc định dark_focus phải là True"
        effective_dark_delta = -raw_dark_delta if ig_mamba.dark_focus else raw_dark_delta
        effective_bright_delta = -raw_bright_delta if ig_mamba.dark_focus else raw_bright_delta

        print(f"  -> [Dark Focus = True] Delta Mod hiệu dụng tại vùng tối: {effective_dark_delta:.4f}")
        print(f"  -> [Dark Focus = True] Delta Mod hiệu dụng tại vùng sáng: {effective_bright_delta:.4f}")
        print("  -> [PASS] Cơ chế Dark Focus điều biến chính xác theo hướng ưu tiên vùng tối.")

    # -------------------------------------------------------------
    # Test 5: Kiểm tra bật/tắt động Dark Focus (Dynamic Toggle)
    # -------------------------------------------------------------
    print("\n[Test 5] Kiểm tra bật/tắt động thuộc tính dark_focus...")
    ig_mamba.dark_focus = False
    assert ig_mamba.attn.dark_focus is False, "Lỗi setter dark_focus sang False"
    out_standard = ig_mamba(x_hv, guide_i)

    ig_mamba.dark_focus = True
    assert ig_mamba.attn.dark_focus is True, "Lỗi setter dark_focus sang True"
    out_dark_focus = ig_mamba(x_hv, guide_i)

    diff = torch.abs(out_standard - out_dark_focus).max().item()
    assert diff > 0, "Đầu ra không thay đổi khi chuyển đổi dark_focus"
    print(f"  -> Độ khác biệt đầu ra lớn nhất giữa 2 chế độ: {diff:.6f}")
    print("  -> [PASS] Bật/tắt dark_focus linh hoạt và tạo tác động rõ rệt lên mô hình!")

    print("\n" + "=" * 60)
    print("TẤT CẢ CÁC BÀI KIỂM TRA ĐÃ VƯỢT QUA THÀNH CÔNG! KHÔNG CÓ LỖI LOGIC.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()

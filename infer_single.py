import os
import sys
import time
import argparse
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

# Đảm bảo console Windows hỗ trợ UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Đảm bảo có thể import các module nội bộ từ thư mục gốc
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

def load_model(weight_path, device, lol=False, v2=False, unpaired=False, alpha=1.0, alpha_s=1.3):
    """
    Khởi tạo mô hình CIDNet và nạp checkpoint weights.
    Tự động phát hiện kiến trúc mô hình (có hoặc không có EdgeLCA) dựa vào checkpoint.
    Cấu hình cờ gated / alpha tương tự như trong eval.py.
    """
    if not os.path.isfile(weight_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số: {weight_path}")

    print(f"[*] Đang tải weights từ: {weight_path}")
    state_dict = torch.load(weight_path, map_location=device)
    
    # Hỗ trợ cả trường hợp checkpoint chứa 'state_dict' hoặc 'model' key
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    # Tự động nhận diện biến thể mô hình: có EdgeLCA hay không
    has_edge = any('edge_extractor' in k for k in state_dict.keys())
    if has_edge:
        print("[*] Nhận diện biến thể: IGMamba + EdgeLCA (CIDNet_Mamba_separable_learning_edge)")
        from net.CIDNet_Mamba_separable_learning_edge import CIDNet
    else:
        print("[*] Nhận diện biến thể: tiêu chuẩn (CIDNet base)")
        from net.CIDNet_base import CIDNet

    model = CIDNet().to(device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # Thiết lập cơ chế gated tương tự eval.py
    if hasattr(model, 'trans'):
        if lol:
            model.trans.gated = True
            model.trans.alpha_s = alpha_s
            print(f"[*] Chế độ LOLv1: gated=True (alpha_s={alpha_s})")
        elif v2:
            model.trans.gated2 = True
            model.trans.alpha = alpha
            print(f"[*] Chế độ LOLv2: gated2=True, alpha={alpha}")
        elif unpaired:
            model.trans.gated2 = True
            model.trans.alpha = alpha
            print(f"[*] Chế độ Unpaired: gated2=True, alpha={alpha}")
        else:
            # Nếu người dùng truyền custom alpha/alpha_s mà không chọn preset
            if alpha != 1.0:
                model.trans.gated2 = True
                model.trans.alpha = alpha
                print(f"[*] Tuỳ chỉnh: gated2=True, alpha={alpha}")
            if alpha_s != 1.3:
                model.trans.gated = True
                model.trans.alpha_s = alpha_s
                print(f"[*] Tuỳ chỉnh: gated=True, alpha_s={alpha_s}")

    return model


def infer_single_image(model, input_path, output_path, device, gamma=1.0, pad_factor=8):
    """
    Thực hiện tăng cường sáng cho 1 ảnh đơn lẻ:
    1. Đọc ảnh RGB
    2. Reflection pad để kích thước chia hết cho pad_factor (mặc định là 8 do mạng có 3 lần downsample)
    3. Feed qua mô hình với input ** gamma
    4. Cắt bỏ padding (crop về kích thước gốc)
    5. Clamp [0, 1] và lưu ảnh ra file
    """
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh đầu vào: {input_path}")

    print(f"[*] Đang đọc ảnh: {input_path}")
    img = Image.open(input_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f"[*] Kích thước gốc: {orig_w}x{orig_h} (Width x Height)")

    # Chuyển thành Tensor [1, 3, H, W] trong dải [0, 1]
    tensor_in = transforms.ToTensor()(img).unsqueeze(0).to(device)

    # Padding kích thước chia hết cho pad_factor (tương tự DatasetFromFolderEval / SICE trong data)
    h, w = orig_h, orig_w
    H = ((h + pad_factor - 1) // pad_factor) * pad_factor
    W = ((w + pad_factor - 1) // pad_factor) * pad_factor
    pad_h = H - h
    pad_w = W - w

    if pad_h > 0 or pad_w > 0:
        # pad thứ tự: (pad_left, pad_right, pad_top, pad_bottom)
        tensor_in = F.pad(tensor_in, (0, pad_w, 0, pad_h), mode='reflect')
        print(f"[*] Đã pad ảnh lên kích thước: {W}x{H} (bội số của {pad_factor})")

    start_time = time.time()
    with torch.no_grad():
        # gamma correction cho đầu vào tương tự eval.py
        model_input = tensor_in ** gamma
        output = model(model_input)
        output = torch.clamp(output, 0.0, 1.0)

        # Cắt về kích thước gốc nếu đã pad
        if pad_h > 0 or pad_w > 0:
            output = output[:, :, :h, :w]

    elapsed = (time.time() - start_time) * 1000
    print(f"[*] Thời gian xử lý mạng (Inference time): {elapsed:.2f} ms")

    # Lưu ảnh kết quả
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    output_pil = transforms.ToPILImage()(output.squeeze(0).cpu())
    output_pil.save(output_path)
    print(f"[✓] Ảnh kết quả đã được lưu tại: {output_path}")

    if device.type == 'cuda':
        torch.cuda.empty_cache()

    return output_pil


def main():
    parser = argparse.ArgumentParser(description='Inference 1 ảnh đơn lẻ với mô hình IGMamba / CIDNet LLIE')
    
    # Tham số đầu vào & đầu ra
    parser.add_argument('--input', '-i', type=str, required=True, help='Đường dẫn đến file ảnh tối cần xử lý')
    parser.add_argument('--output', '-o', type=str, default=None, help='Đường dẫn file ảnh đầu ra (mặc định lưu tại ./output/<tên_gốc>_enhanced.png)')
    parser.add_argument('--weight', '-w', type=str, default=None, help='Đường dẫn file trọng số .pth')

    # Preset trọng số và chế độ theo bộ dữ liệu (tương tự eval.py)
    parser.add_argument('--lol', action='store_true', help='Preset bộ dữ liệu LOLv1 (gated=True)')
    parser.add_argument('--lol_v2_real', action='store_true', help='Preset bộ dữ liệu LOLv2-Real (gated2=True, alpha=0.8)')
    parser.add_argument('--lol_v2_syn', action='store_true', help='Preset bộ dữ liệu LOLv2-Synthetic (gated2=True)')
    parser.add_argument('--unpaired', action='store_true', help='Preset ảnh chụp thực tế không có cặp / Unpaired (gated2=True)')

    # Tuỳ chỉnh tham số nâng cao
    parser.add_argument('--alpha', type=float, default=None, help='Hệ số khuếch đại ánh sáng alpha (mặc định: 0.8 cho lol_v2_real, 1.0 cho chế độ khác)')
    parser.add_argument('--alpha_s', type=float, default=1.3, help='Hệ số bão hòa màu alpha_s (mặc định: 1.3)')
    parser.add_argument('--gamma', type=float, default=1.0, help='Hệ số gamma cho ảnh đầu vào (input ** gamma, mặc định: 1.0)')
    parser.add_argument('--pad_factor', type=int, default=8, help='Bội số padding kích thước ảnh (mặc định: 8)')
    parser.add_argument('--cpu', action='store_true', help='Ép chạy trên CPU ngay cả khi có GPU CUDA')

    args = parser.parse_args()

    # 1. Chọn Device
    if args.cpu or not torch.cuda.is_available():
        device = torch.device('cpu')
        print("[*] Sử dụng thiết bị: CPU")
    else:
        device = torch.device('cuda')
        print(f"[*] Sử dụng thiết bị: GPU ({torch.cuda.get_device_name(0)})")

    # 2. Xác định file trọng số & tham số mặc định dựa vào preset (giống eval.py)
    weight_path = args.weight
    alpha = args.alpha

    if args.lol:
        if weight_path is None:
            weight_path = os.path.join(current_dir, 'pretrained_models', 'LOLv1', 'best_PSNR.pth')
        if alpha is None:
            alpha = 1.0
    elif args.lol_v2_real:
        if weight_path is None:
            weight_path = os.path.join(current_dir, 'pretrained_models', 'LOLv2_Real', 'best_PSNR.pth')
        if alpha is None:
            alpha = 0.8  # Giá trị tối ưu cho LOLv2-Real trong eval.py
    elif args.lol_v2_syn:
        if weight_path is None:
            weight_path = os.path.join(current_dir, 'pretrained_models', 'LOLv2_Syn', 'best_PSNR.pth')
        if alpha is None:
            alpha = 1.0
    elif args.unpaired:
        if weight_path is None:
            weight_path = os.path.join(current_dir, 'pretrained_models', 'Unpaired', 'best_NIQE.pth')
        if alpha is None:
            alpha = 1.0
    else:
        # Mặc định nếu không chỉ định preset cụ thể
        if weight_path is None:
            # Ưu tiên LOLv1 best_PSNR hoặc file có sẵn trong repo
            candidate_weights = [
                os.path.join(current_dir, 'pretrained_models', 'LOLv1', 'best_PSNR.pth'),
                os.path.join(current_dir, 'pretrained_models', 'LOLv2_Real', 'best_PSNR.pth'),
                os.path.join(current_dir, 'weights', 'LOLv1', 'w_perc.pth'),
            ]
            for cw in candidate_weights:
                if os.path.isfile(cw):
                    weight_path = cw
                    break
            if weight_path is None:
                raise ValueError("Vui lòng chỉ định đường dẫn file trọng số qua tham số --weight <đường_dẫn>")
        if alpha is None:
            alpha = 1.0

    # 3. Xác định đường dẫn file đầu ra
    output_path = args.output
    if output_path is None:
        base_name = os.path.basename(args.input)
        name_without_ext, ext = os.path.splitext(base_name)
        if not ext:
            ext = '.png'
        output_path = os.path.join(current_dir, 'output', f"{name_without_ext}_enhanced{ext}")

    # 4. Load model
    model = load_model(
        weight_path=weight_path,
        device=device,
        lol=args.lol,
        v2=args.lol_v2_real,
        unpaired=args.unpaired,
        alpha=alpha,
        alpha_s=args.alpha_s
    )

    # 5. Chạy inference
    infer_single_image(
        model=model,
        input_path=args.input,
        output_path=output_path,
        device=device,
        gamma=args.gamma,
        pad_factor=args.pad_factor
    )


if __name__ == '__main__':
    main()

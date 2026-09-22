import os
import sys
import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Thêm đường dẫn thư mục gốc HVI-CIDNet để import được module net
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

from net.HVI_transform import RGB_HVI


def convert_to_hvi_and_save(image_path, output_dir, save_separate=False):
    """
    Chuyển đổi ảnh RGB sang không gian màu HVI và lưu kết quả:
    - HV.png: Ảnh gộp 2 chiều màu sắc H và V (Kênh R=H_norm, G=V_norm, B=Magnitude/Saturation)
    - HV_color.png: Ảnh màu sắc thuần túy tái tạo từ HV khi chuẩn hóa độ sáng I = 1.0 (Pure Chrominance)
    - I.png: Ảnh kênh độ rọi/ánh sáng (Grayscale)
    - (Tùy chọn) H.png, V.png nếu save_separate=True
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Đọc ảnh RGB đầu vào
    img = Image.open(image_path).convert('RGB')
    img_tensor = TF.to_tensor(img).unsqueeze(0)  # [1, 3, H, W] trong khoảng [0, 1]
    
    # 2. Khởi tạo bộ biến đổi HVI
    converter = RGB_HVI()
    
    with torch.no_grad():
        # Chuyển đổi sang HVI: H, V in [-1, 1], I in [0, 1]
        hvi_tensor = converter.HVIT(img_tensor)
        
        H = hvi_tensor[:, 0:1, :, :]  # [1, 1, H, W]
        V = hvi_tensor[:, 1:2, :, :]  # [1, 1, H, W]
        I = hvi_tensor[:, 2:3, :, :]  # [1, 1, H, W]
        
        # 3. GỘP HV LẠI THÀNH ẢNH MÀU:
        # H và V có dải giá trị [-1, 1], đưa về [0, 1] để hiển thị
        H_vis = (H + 1.0) / 2.0
        V_vis = (V + 1.0) / 2.0
        
        # Kênh B: Độ lớn vector màu (Magnitude / Saturation)
        mag = torch.sqrt(H**2 + V**2)
        mag_vis = torch.clamp(mag, 0.0, 1.0)
        
        # Tạo ảnh gộp HV dạng False-color (R=H, G=V, B=Magnitude)
        HV_tensor = torch.cat([H_vis, V_vis, mag_vis], dim=1)
        HV_img = TF.to_pil_image(HV_tensor.squeeze(0))
        
        # Tạo ảnh màu sắc tái tạo thực tế (Pure Chrominance Map khi cố định I = 1.0)
        # Giúp quan sát thông tin màu thuần túy mà không bị tối
        hvi_flat_i = torch.cat([H, V, torch.ones_like(I)], dim=1)
        pure_color_tensor = converter.PHVIT(hvi_flat_i)
        pure_color_tensor = torch.clamp(pure_color_tensor, 0.0, 1.0)
        HV_color_img = TF.to_pil_image(pure_color_tensor.squeeze(0))
        
        # 4. Kênh độ rọi I (ảnh xám 3 kênh RGB)
        I_vis = torch.clamp(I, 0.0, 1.0)
        I_tensor = torch.cat([I_vis, I_vis, I_vis], dim=1)
        I_img = TF.to_pil_image(I_tensor.squeeze(0))
        
        # 5. Lưu các ảnh chính
        hv_path = os.path.join(output_dir, 'HV.png')
        hv_color_path = os.path.join(output_dir, 'HV_color.png')
        i_path = os.path.join(output_dir, 'I.png')
        orig_path = os.path.join(output_dir, 'original.png')
        
        img.save(orig_path)
        HV_img.save(hv_path)
        HV_color_img.save(hv_color_path)
        I_img.save(i_path)
        
        print(f"-> [Saved] Original RGB:       {orig_path}")
        print(f"-> [Saved] Merged HV (Vector):  {hv_path}")
        print(f"-> [Saved] Merged HV (Color):   {hv_color_path}")
        print(f"-> [Saved] Channel I (Light):   {i_path}")
        
        # 6. Lưu riêng từng kênh H, V nếu người dùng yêu cầu
        if save_separate:
            H_rgb = torch.cat([H_vis, H_vis, H_vis], dim=1)
            V_rgb = torch.cat([V_vis, V_vis, V_vis], dim=1)
            h_path = os.path.join(output_dir, 'H.png')
            v_path = os.path.join(output_dir, 'V.png')
            TF.to_pil_image(H_rgb.squeeze(0)).save(h_path)
            TF.to_pil_image(V_rgb.squeeze(0)).save(v_path)
            print(f"-> [Saved] Individual H:        {h_path}")
            print(f"-> [Saved] Individual V:        {v_path}")


if __name__ == '__main__':
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    import argparse
    parser = argparse.ArgumentParser(description='Convert RGB image to HVI color space with merged HV channels and I channel.')
    parser.add_argument('--input', type=str, required=True, help='Path to input image')
    parser.add_argument('--output_dir', type=str, default='output_hvi', help='Output directory')
    parser.add_argument('--save_separate', action='store_true', help='Also save separate H.png and V.png')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Image not found: {args.input}")
    else:
        convert_to_hvi_and_save(args.input, args.output_dir, save_separate=args.save_separate)

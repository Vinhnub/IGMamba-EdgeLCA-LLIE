import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeExtractor(nn.Module):
    """
    Trích xuất bản đồ cạnh (edge map) từ ảnh đầu vào, TÍCH HỢP TỪ LREMNet.
    Module này bao gồm 3 bước chính:
    1. Làm mờ (Gaussian Blur) để loại bỏ nhiễu hạt (noise).
    2. Trích xuất đa hướng (Sobel X, Sobel Y, Laplacian) và dung hợp qua Conv2d 1x1 (có thể học).
    3. Chuyển đổi thành điểm ưu tiên (Priority Score) qua Sigmoid.
    """
    def __init__(self, in_channels=3, rgb_to_gray=False, use_blur=True, blur_kernel_size=3, blur_sigma=1.0, **kwargs):
        super(EdgeExtractor, self).__init__()
        self.in_channels = in_channels
        self.rgb_to_gray = (rgb_to_gray and in_channels == 3)
        self.use_blur = use_blur
        
        # 1. Bộ lọc Gaussian làm mờ dập tắt nhiễu hạt (Noise Suppression Pre-filter)
        if self.use_blur:
            self.register_buffer('gaussian_kernel', self._create_gaussian_kernel(blur_kernel_size, blur_sigma))

        # 2. Tạo các bộ lọc trích xuất cạnh (Từ LREMNet)
        self.register_buffer('sobel_x', torch.tensor([
            [-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]
        ]).view(1, 1, 3, 3))
        
        self.register_buffer('sobel_y', torch.tensor([
            [-1., -2., -1.], [ 0.,  0.,  0.], [ 1.,  2.,  1.]
        ]).view(1, 1, 3, 3))
        
        self.register_buffer('laplacian', torch.tensor([
            [0., -1., 0.], [-1., 4., -1.], [0., -1., 0.]
        ]).view(1, 1, 3, 3))
        
        # Lớp học dung hợp 3 bộ lọc (Channel Fusion)
        # Khởi tạo có trọng số thay vì bias=False để học tốt hơn hoặc giữ nguyên bias=False như LREMNet gốc
        self.channel_fusion = nn.Conv2d(3, 1, 1, bias=False)
        
        # Các tham số cho Priority Mapping (Tăng scale và giảm offset để cạnh nét hơn, contrast cao hơn)
        self.scale = nn.Parameter(torch.tensor(10.0))
        self.offset = nn.Parameter(torch.tensor(-5.0))

    def _create_gaussian_kernel(self, kernel_size, sigma):
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
        grid_x, grid_y = torch.meshgrid(coords, coords, indexing='ij')
        variance = sigma ** 2.0
        gaussian = torch.exp(-(grid_x**2 + grid_y**2) / (2 * variance))
        gaussian = gaussian / torch.sum(gaussian)
        return gaussian.view(1, 1, kernel_size, kernel_size)

    def forward(self, x, average_channels=False):
        """
        Lưu ý: Do tích hợp LREMNet, đầu ra sẽ luôn được gom về 1 kênh (Bản đồ ưu tiên).
        Tham số average_channels được giữ lại để tương thích ngược với code cũ nhưng không còn tác dụng phụ.
        """
        B, C, H, W = x.shape
        if C != self.in_channels:
            raise ValueError(f"Module khởi tạo với in_channels={self.in_channels}, nhưng nhận được {C} kênh.")
            
        # --- BƯỚC 1: GOM KÊNH ---
        # Đưa về 1 kênh duy nhất trước khi tìm cạnh để dễ áp dụng dung hợp 3 hướng
        if C > 1:
            if self.rgb_to_gray:
                weight = torch.tensor([0.299, 0.587, 0.114], dtype=x.dtype, device=x.device).view(1, C, 1, 1)
                target = torch.sum(x * weight, dim=1, keepdim=True)
            else:
                target = x.mean(dim=1, keepdim=True)
        else:
            target = x

        # --- BƯỚC 2: LÀM MỜ DẬP TẮT NHIỄU HẠT (Gaussian Blur Pre-filter) ---
        # Triệt tiêu các điểm nhiễu cô lập trước khi tính đạo hàm không gian
        if self.use_blur:
            target_blur_pad = F.pad(target, (1, 1, 1, 1), mode='replicate')
            target = F.conv2d(target_blur_pad, self.gaussian_kernel)
            
        # --- BƯỚC 3: TRÍCH XUẤT ĐA HƯỚNG ---
        target_padded = F.pad(target, (1, 1, 1, 1), mode='replicate')
        grad_x = F.conv2d(target_padded, self.sobel_x)
        grad_y = F.conv2d(target_padded, self.sobel_y)
        grad_lap = F.conv2d(target_padded, self.laplacian)
        
        # --- BƯỚC 4: DUNG HỢP (Học Trọng Số) ---
        grad_stack = torch.cat([grad_x, grad_y, grad_lap], dim=1) # [B, 3, H, W]
        gradient_magnitude = self.channel_fusion(grad_stack)      # [B, 1, H, W]
        gradient_magnitude = torch.abs(gradient_magnitude)
        
        # --- BƯỚC 5: TẠO ĐIỂM ƯU TIÊN (Priority Score) ---
        gradient_flat = gradient_magnitude.view(B, -1)
        grad_min = gradient_flat.min(dim=1, keepdim=True)[0]
        grad_max = gradient_flat.max(dim=1, keepdim=True)[0]
        
        # Chuẩn hóa Min-Max về [0, 1]
        gradient_norm = (gradient_flat - grad_min) / (grad_max - grad_min + 1e-8)
        gradient_norm = gradient_norm.view(B, 1, H, W)
        
        # Affine Transform & Sigmoid
        priority_score = self.scale * gradient_norm + self.offset
        edge_map = torch.sigmoid(priority_score)
        
        return edge_map

if __name__ == '__main__':
    import os
    import sys
    # Thêm đường dẫn gốc HVI-CIDNet để có thể import HVI_transform
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    sys.path.append(parent_dir)
    
    from net.HVI_transform import RGB_HVI
    import torchvision.transforms as T
    from torchvision.utils import save_image
    from PIL import Image

    print("=== TEST EDGE EXTRACTOR ===")
    # Tìm một ảnh mẫu trong thư mục HVI-CIDNet
    img_path = os.path.join(parent_dir, r'E:\PythonFile\Project\Low-Light-Image-Enhancement\mydata\dataset\dataset\LOLv1\test\high\778.png')
    
    if os.path.exists(img_path):
        print(f"Đang đọc ảnh: {img_path}")
        # Đọc ảnh RGB
        img = Image.open(img_path).convert('RGB')
        # Chuyển thành tensor [1, 3, H, W] trong khoảng [0, 1]
        x = T.ToTensor()(img).unsqueeze(0)
        
        # 1. Khởi tạo bộ chuyển đổi HVI
        hvi_trans = RGB_HVI()
        hvi = hvi_trans.HVIT(x)
        print("Kích thước HVI:", hvi.shape)
        
        # 2. Tách thành các nhánh để thử nghiệm
        hv = hvi[:, 0:2, :, :] # 2 kênh H và V
        i = hvi[:, 2:3, :, :]  # kênh cường độ sáng I
        
        # 3. Thử nghiệm trên nhánh HV (2 kênh)
        hv_extractor = EdgeExtractor(in_channels=2, blur_kernel_size=5, blur_sigma=1.5)
        # Gộp thành 1 ảnh xám cạnh bằng average_channels=True
        edge_hv_avg = hv_extractor(hv, average_channels=True)
        # Hoặc giữ nguyên 2 kênh cạnh
        edge_hv_2ch = hv_extractor(hv, average_channels=False)
        
        # 4. Thử nghiệm trên nhánh I (1 kênh)
        i_extractor = EdgeExtractor(in_channels=1, blur_kernel_size=3, blur_sigma=0.5)
        edge_i = i_extractor(i)
        
        # 5. Lưu ảnh kết quả ra thư mục output_test_edges
        out_dir = os.path.join(parent_dir, 'output_test_edges')
        os.makedirs(out_dir, exist_ok=True)
        
        # Normalize=True giúp stretch giá trị pixel ra khoảng [0, 1] để dễ nhìn
        save_image(x, os.path.join(out_dir, '0_original.png'))
        save_image(edge_hv_avg, os.path.join(out_dir, '1_edge_HV_avg.png'), normalize=True)
        save_image(edge_i, os.path.join(out_dir, '2_edge_I.png'), normalize=True)
        
        print(f"Đã trích xuất xong! Các ảnh kết quả được lưu tại: {out_dir}")
    else:
        print(f"Không tìm thấy ảnh mẫu {img_path}.")

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_comparison_grid(image_paths, labels, rows=2, cols=5, crop_boxes=None, show_zooms=None, target_w=512, output_path="comparison_grid.jpg", quality_dpi=150):
    """
    Tạo ảnh lưới so sánh tùy chỉnh số hàng/cột (VD: 1x5 hoặc 2x5).
    Mỗi hàng có thể cấu hình tọa độ vùng crop_box riêng, và tự động giữ nguyên tỷ lệ kích thước gốc của ảnh (tránh méo).
    Có thể bật/tắt hiển thị khung zoom cho từng hàng với show_zooms.
    DPI có thể hạ thấp (VD: 100-150) để giảm dung lượng file khi nộp báo cáo, Overleaf.
    """
    # Nếu không truyền crop_boxes, tạo mặc định cho mỗi hàng
    if crop_boxes is None or len(crop_boxes) < rows:
        crop_boxes = [(150, 50, 150, 70)] * rows
        
    # Nếu không truyền show_zooms, mặc định bật zoom cho tất cả các hàng
    if show_zooms is None or len(show_zooms) < rows:
        show_zooms = [True] * rows

    # --- BƯỚC 1: Tính toán chiều cao (pixel) của từng hàng để set height_ratios ---
    row_pixel_heights = []
    processed_images_info = [] # Lưu tạm để vòng lặp sau không phải load lại
    
    for r in range(rows):
        # Lấy ảnh đầu tiên của hàng r để tính tỷ lệ kích thước cho cả hàng
        first_img_idx = r * cols
        if first_img_idx < len(image_paths) and os.path.exists(image_paths[first_img_idx]):
            temp_img = cv2.imread(image_paths[first_img_idx])
            orig_h, orig_w = temp_img.shape[:2]
        else:
            orig_h, orig_w = 725, 750
            
        # Tính target_h để giữ đúng tỷ lệ ảnh gốc (Aspect Ratio)
        target_h = int(target_w * (orig_h / float(orig_w)))
        
        cx, cy, cw, ch = crop_boxes[r]
        
        # Nếu bật zoom thì tính thêm khoảng chiều cao phụ để ghép ảnh zoom vào dưới
        if show_zooms[r]:
            crop_display_h = int(target_w / (cw / ch))
            cell_h = target_h + 4 + crop_display_h
        else:
            crop_display_h = 0
            cell_h = target_h
            
        row_pixel_heights.append(cell_h)
        processed_images_info.append((target_h, cx, cy, cw, ch, crop_display_h, show_zooms[r]))

    # --- BƯỚC 2: Tính tổng kích thước khung hình (Figure) ---
    total_w_px = cols * target_w
    # hspace=0.12 nghĩa là thêm 12% chiều cao của trung bình 1 hàng làm khe hở.
    hspace = 0.12 if rows > 1 else 0.0
    total_h_px = sum(row_pixel_heights) + (rows - 1) * hspace * (sum(row_pixel_heights)/rows)
    
    fig_width = 20.0
    fig_height = fig_width * (total_h_px / total_w_px)

    # Khởi tạo figure với gridspec để phân bổ chiều cao khác nhau cho từng hàng
    gridspec_kw = {'height_ratios': row_pixel_heights} if rows > 1 else None
    fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), gridspec_kw=gridspec_kw)
    
    # Ép axes về mảng 1D để dễ duyệt
    if rows == 1 or cols == 1:
        axes = np.array(axes).flatten()
    else:
        axes = axes.flatten()
        
    plt.subplots_adjust(wspace=0.0, hspace=hspace, top=0.98, bottom=0.02, left=0.0, right=1.0)

    # --- BƯỚC 3: Xử lý và vẽ từng ảnh ---
    for idx, ax in enumerate(axes):
        r = idx // cols  # Xác định đang ở hàng nào
        target_h, cx, cy, cw, ch, crop_display_h, show_zoom = processed_images_info[r]

        # Load ảnh
        if idx < len(image_paths) and os.path.exists(image_paths[idx]):
            img = cv2.imread(image_paths[idx])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.ones((target_h, target_w, 3), dtype=np.uint8) * (50 + (idx%10) * 20)
            cv2.putText(img, f"No Image {idx}", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        img = cv2.resize(img, (target_w, target_h))

        # Nếu hàng này bật zoom, vẽ khung đỏ và ghép ảnh crop bên dưới
        if show_zoom:
            crop_y1, crop_y2 = max(0, cy), min(target_h, cy + ch)
            crop_x1, crop_x2 = max(0, cx), min(target_w, cx + cw)
            crop_img = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            # Vẽ khung đỏ lên ảnh gốc
            cv2.rectangle(img, (crop_x1, crop_y1), (crop_x2, crop_y2), (255, 0, 0), 3)

            # Resize vùng crop bằng INTER_CUBIC cho sắc nét
            crop_img_resized = cv2.resize(crop_img, (target_w, crop_display_h), interpolation=cv2.INTER_CUBIC)

            # Thêm đường viền trắng (hoặc đen) phân cách
            separator = np.ones((4, target_w, 3), dtype=np.uint8) * 255

            # Ghép ảnh gốc, viền phân cách, và ảnh crop
            combined_img = np.vstack((img, separator, crop_img_resized))
        else:
            combined_img = img

        # Hiển thị lên axes
        ax.imshow(combined_img, aspect='auto')
        ax.axis('off')

        # Thêm nhãn
        label = labels[idx] if idx < len(labels) else f"({chr(97+idx)})"
        ax.text(0.5, -0.05, label, 
                transform=ax.transAxes, 
                ha='center', va='top', 
                fontsize=16, fontweight='bold', color='black')

    plt.savefig(output_path, dpi=quality_dpi, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    print(f"Đã lưu bảng so sánh tại: {output_path} (DPI={quality_dpi})")

if __name__ == "__main__":
    # --- CẤU HÌNH ĐƯỜNG DẪN ẢNH VÀ NHÃN ---
    image_list = [
        r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\dataset\LOL\LOLv2-real\Test\Input\00768.png", # (a) Input
        r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\output_other_model\KinD\LOLv2_real\00768.png",                    # (b) Model 1
        r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\output_other_model\SNR\LOLv2_real\00768.png",                   # (c) Model 2
        r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\output_other_model\RetinexFormer\LOLv2_real\00768.png",                   # (d) Model 3
        r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\output_other_model\RetinexMamba\LOLv2_real\00768.png",                   # (e) Model 4
        
        r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\output_other_model\CWNet\LOLv2_real\00768.png",                     # (f) Input
        r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\output_other_model\CIDNet\LOLv2_real\00768.png",                    # (g) Model 1
        r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\output_other_model\InterLight\LOLv2_real\00768.png",                   # (h) Model 2
        r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\output_other_model\Our\LOLv2_real\00768.png",                   # (i) Model 3
        r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\dataset\LOL\LOLv2-real\Test\GT\00768.png",       # (j) GT
    ]

    model_labels = [
        "(a) Input", "(b) KinD", "(c) SNR-Aware", "(d) RetinexFormer", "(e) RetinexMamba",
        "(f) CWNet", "(g) CIDNet", "(h) InterLight", "(i) Our Method", "(j) Ground Truth",
    ]

    # Khai báo riêng vùng crop cho từng hàng (x, y, w, h)
    crop_row_1 = (100, 200, 150, 70) # Tọa độ crop cho bộ ảnh hàng trên (Balloons)
    crop_row_2 = (100, 200, 150, 70)  # Tọa độ crop cho bộ ảnh hàng dưới (BelgiumHouse)
    
    os.makedirs("visualization", exist_ok=True)
    
    # 1. TẠO BẢNG 1x5 (Chỉ lấy 5 ảnh đầu)
    # Tùy chỉnh: quality_dpi hạ xuống 150 để giảm dung lượng file khi import Overleaf
    # Lưu dưới đuôi .jpg thay vì .png để siêu nhẹ!
    create_comparison_grid(
        image_list[:5], 
        model_labels[:5], 
        rows=1, cols=5, 
        crop_boxes=[crop_row_1], 
        show_zooms=[True], 
        quality_dpi=150,
        output_path="visualization/comparison_grid_1x5.jpg"
    )

    # 2. TẠO BẢNG 2x5 (Lấy cả 10 ảnh)
    # Hàng 1 Tắt zoom, Hàng 2 Bật zoom
    create_comparison_grid(
        image_list, 
        model_labels, 
        rows=2, cols=5, 
        crop_boxes=[crop_row_1, crop_row_2], 
        show_zooms=[True, True],
        quality_dpi=150,
        output_path="visualization/comparison_grid_2x5.jpg"
    )

import os
import sys
import glob
import re
import argparse
from typing import List, Dict, Tuple, Union, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Đảm bảo in console UTF-8 an toàn trên Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Cấu hình thẩm mỹ cho matplotlib
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def canonicalize_col_name(name: str) -> str:
    """Chuẩn hóa tên cột từ file markdown."""
    clean = name.strip().lower()
    clean = re.sub(r'[\s_]+', '_', clean)
    
    if 'epoch' in clean:
        return 'epoch'
    if 'total' in clean and 'loss' in clean:
        return 'total_loss'
    if clean == 'l1':
        return 'l1'
    if clean == 'l2':
        return 'l2'
    if 'ssim' in clean and ('d' in clean or 'd(' in clean):
        return 'd_ssim'
    if 'vgg' in clean or 'p(' in clean:
        return 'p_vgg'
    if 'edge' in clean:
        return 'edge'
    if 'lsgd' in clean:
        return 'lsgd'
    if 'exp' in clean:
        return 'exp'
    if 'psnr' in clean:
        return 'psnr'
    if 'ssim' in clean:
        return 'ssim'
    if 'lpips' in clean:
        return 'lpips'
    if 'note' in clean:
        return 'note'
    return clean


def parse_log_files(
    file_inputs: Union[str, List[str]]
) -> Tuple[Dict[int, Dict[str, float]], Dict[str, str], List[str]]:
    """
    Đọc một hoặc nhiều file log markdown (metrics*.md).
    - Tự động bỏ qua toàn bộ dòng đánh giá GT Mean.
    - Chỉ giữ lại metric chuẩn/gốc của model (Standard / Direct evaluation).
    - Tự động nối và hợp nhất các file log khi train bị ngắt quãng / chia nhỏ nhiều lần.
    - Sắp xếp tăng dần theo Epoch và khử trùng lặp dữ liệu (ghi đè checkpoint sau).
    """
    file_paths: List[str] = []
    
    # 1. Phân giải danh sách file đầu vào
    if isinstance(file_inputs, str):
        if '*' in file_inputs or '?' in file_inputs:
            file_paths = sorted(glob.glob(file_inputs))
        elif os.path.isdir(file_inputs):
            file_paths = sorted(glob.glob(os.path.join(file_inputs, "metrics*.md")))
        else:
            file_paths = [file_inputs]
    elif isinstance(file_inputs, (list, tuple)):
        for item in file_inputs:
            if '*' in item or '?' in item:
                file_paths.extend(sorted(glob.glob(item)))
            elif os.path.isdir(item):
                file_paths.extend(sorted(glob.glob(os.path.join(item, "metrics*.md"))))
            else:
                file_paths.append(item)
    
    valid_files = [f for f in file_paths if os.path.isfile(f)]
    if not valid_files:
        raise FileNotFoundError(f"Không tìm thấy file log nào hợp lệ từ: {file_inputs}")
        
    metadata: Dict[str, str] = {}
    data: Dict[int, Dict[str, float]] = {}

    for fpath in valid_files:
        header_keys: Optional[List[str]] = None
        
        with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if not stripped:
                    continue
                
                # Trích xuất config ở đầu file (dataset, lr, batch size, v.v.)
                if ":" in stripped and not stripped.startswith("|"):
                    parts = stripped.split(":", 1)
                    key = parts[0].strip()
                    val = parts[1].strip()
                    if key not in metadata:
                        metadata[key] = val
                    continue
                
                if not stripped.startswith("|"):
                    continue
                
                # Cắt các ô trong dòng bảng markdown
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if not cells:
                    continue
                
                # Bỏ qua dòng kẻ phân cách markdown (|---|---|)
                if any(c.startswith("---") or c.startswith(":-") for c in cells):
                    continue
                
                # Dòng tiêu đề cột
                if any("epoch" in c.lower() for c in cells):
                    header_keys = [canonicalize_col_name(c) for c in cells]
                    continue
                
                # BỎ QUA HOÀN TOÀN CÁC DÒNG CÓ GT MEAN
                if any("gt" in c.lower() for c in cells[1:]):
                    continue
                
                # Kiểm tra ô đầu tiên có phải epoch số nguyên hay không
                try:
                    epoch = int(cells[0])
                except ValueError:
                    continue
                
                # Ánh xạ tên cột vào giá trị
                row_dict: Dict[str, float] = {"epoch": epoch}
                
                if header_keys and len(header_keys) == len(cells):
                    for h_name, cell_val in zip(header_keys, cells):
                        if h_name in ('epoch', 'note'):
                            continue
                        try:
                            row_dict[h_name] = float(cell_val)
                        except ValueError:
                            pass
                else:
                    # Trường hợp file nối tiếp không có dòng header:
                    # - Định dạng chỉ có metrics: [Epoch, PSNR, SSIM, LPIPS]
                    if len(cells) in (4, 5):
                        names = ["epoch", "psnr", "ssim", "lpips"]
                        for i, name in enumerate(names):
                            if i < len(cells):
                                try:
                                    row_dict[name] = float(cells[i])
                                except ValueError:
                                    pass
                    # - Định dạng đầy đủ có loss: [Epoch, Total Loss, L1, L2, D, P, Edge, LSGD, EXP, PSNR, SSIM, LPIPS, Note]
                    elif len(cells) >= 12:
                        names = [
                            "epoch", "total_loss", "l1", "l2", "d_ssim", "p_vgg",
                            "edge", "lsgd", "exp", "psnr", "ssim", "lpips"
                        ]
                        for i, name in enumerate(names):
                            if i < len(cells):
                                try:
                                    row_dict[name] = float(cells[i])
                                except ValueError:
                                    pass
                
                # Cập nhật / ghi đè dữ liệu theo từng epoch (tự động xử lý overlap khi resume checkpoint)
                if epoch not in data:
                    data[epoch] = {}
                data[epoch].update(row_dict)

    return data, metadata, valid_files


def extract_series(
    data_dict: Dict[int, Dict[str, float]], 
    metric_name: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Trích xuất mảng epoch và mảng giá trị cho một metric cụ thể, đã lọc sạch NaN/None."""
    sorted_epochs = sorted(data_dict.keys())
    valid_epochs = []
    values = []
    for ep in sorted_epochs:
        row = data_dict[ep]
        if metric_name in row:
            valid_epochs.append(ep)
            values.append(row[metric_name])
    return np.array(valid_epochs, dtype=int), np.array(values, dtype=float)


def print_summary_table(
    data: Dict[int, Dict[str, float]], 
    metadata: Dict[str, str],
    valid_files: List[str]
):
    """In bảng tổng kết các chỉ số tốt nhất đạt được."""
    print("\n" + "=" * 76)
    print("                 TRAINING SUMMARY & BEST METRICS (STANDARD)")
    print("=" * 76)
    
    if metadata:
        print("Detected Training Configuration:")
        for k, v in metadata.items():
            print(f"  - {k:<15}: {v}")
        print("-" * 76)
        
    all_epochs = sorted(data.keys())
    if not all_epochs:
        print("No valid epochs found!")
        print("=" * 76 + "\n")
        return
        
    print(f"Log files processed: {len(valid_files)}")
    for f in valid_files:
        print(f"  * {f}")
    print(f"Total evaluated epochs: {len(all_epochs)} (Epoch {min(all_epochs)} -> {max(all_epochs)})")
    print("-" * 76)
    print(f"{'Metric':<20} | {'Best Value':<18} | {'At Epoch':<10} | {'Remark':<18}")
    print("-" * 76)

    # 1. PSNR (Higher is better)
    ep_psnr, val_psnr = extract_series(data, "psnr")
    if len(val_psnr) > 0:
        idx = np.argmax(val_psnr)
        print(f"{'PSNR':<20} | {val_psnr[idx]:.4f} dB{'':<9} | {ep_psnr[idx]:<10} | {'Max (Higher better)':<18}")

    # 2. SSIM (Higher is better)
    ep_ssim, val_ssim = extract_series(data, "ssim")
    if len(val_ssim) > 0:
        idx = np.argmax(val_ssim)
        print(f"{'SSIM':<20} | {val_ssim[idx]:.4f}{'':<12} | {ep_ssim[idx]:<10} | {'Max (Higher better)':<18}")

    # 3. LPIPS (Lower is better)
    ep_lpips, val_lpips = extract_series(data, "lpips")
    if len(val_lpips) > 0:
        idx = np.argmin(val_lpips)
        print(f"{'LPIPS':<20} | {val_lpips[idx]:.4f}{'':<12} | {ep_lpips[idx]:<10} | {'Min (Lower better)':<18}")

    # 4. Total Loss nếu có
    ep_loss, val_loss = extract_series(data, "total_loss")
    if len(val_loss) > 0:
        idx = np.argmin(val_loss)
        print(f"{'Total Loss':<20} | {val_loss[idx]:.4f}{'':<12} | {ep_loss[idx]:<10} | {'Min Training Loss':<18}")

    print("=" * 76 + "\n")


def plot_learning_curves(
    data: Dict[int, Dict[str, float]], 
    save_dir: str,
    prefix: str = "learning_curve"
):
    """
    Vẽ 3 metrics (PSNR, SSIM, LPIPS) chung vào 1 biểu đồ duy nhất (sử dụng 3 trục Y),
    thiết kế chống chồng lấn hoàn toàn (Non-overlapping layout) và lưu thành file .jpg.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    ep_psnr, val_psnr = extract_series(data, "psnr")
    ep_ssim, val_ssim = extract_series(data, "ssim")
    ep_lpips, val_lpips = extract_series(data, "lpips")

    if len(val_psnr) == 0 and len(val_ssim) == 0 and len(val_lpips) == 0:
        print("Cảnh báo: Không có dữ liệu PSNR, SSIM hoặc LPIPS để vẽ.")
        return

    # -------------------------------------------------------------------------
    # VẼ CẢ 3 METRICS VÀO 1 BIỂU ĐỒ DUY NHẤT (MULTI-AXIS: 3 TRỤC Y)
    # -------------------------------------------------------------------------
    fig, ax1 = plt.subplots(figsize=(13.0, 6.8), dpi=200)

    lines = []
    labels = []

    # Màu sắc phong cách báo chí khoa học
    c_psnr = "#1f77b4"     # Xanh lam đậm
    c_ssim = "#2ca02c"     # Xanh lá cây
    c_lpips = "#d62728"    # Đỏ tươi

    # Tính toán chỉ số tốt nhất của từng metric
    best_p_ep, best_p_val = None, None
    best_s_ep, best_s_val = None, None
    best_l_ep, best_l_val = None, None

    all_epochs = []
    if len(val_psnr) > 0:
        idx = np.argmax(val_psnr)
        best_p_ep, best_p_val = int(ep_psnr[idx]), float(val_psnr[idx])
        all_epochs.extend(ep_psnr)
    if len(val_ssim) > 0:
        idx = np.argmax(val_ssim)
        best_s_ep, best_s_val = int(ep_ssim[idx]), float(val_ssim[idx])
        all_epochs.extend(ep_ssim)
    if len(val_lpips) > 0:
        idx = np.argmin(val_lpips)
        best_l_ep, best_l_val = int(ep_lpips[idx]), float(val_lpips[idx])
        all_epochs.extend(ep_lpips)

    max_epoch = max(all_epochs) if all_epochs else 1000

    # Tính toán vị trí offset động thông minh để các hộp chú thích không bao giờ đè lên nhau
    # 1. PSNR offset
    if best_p_ep is not None and best_s_ep is not None and best_p_ep <= best_s_ep:
        p_offset = (-45, 22)
    else:
        p_offset = (20, 24)

    # 2. SSIM offset (nếu gần PSNR thì đẩy xuống dưới để tách biệt)
    if best_p_ep is not None and best_s_ep is not None and abs(best_p_ep - best_s_ep) <= 45:
        s_offset = (-50, -28)
    elif best_p_ep is not None and best_s_ep is not None and best_s_ep > best_p_ep:
        s_offset = (20, 22)
    else:
        s_offset = (-45, 22)

    # 3. LPIPS offset (nếu ở sát mép phải thì lùi về trái tránh đè trục Y)
    if best_l_ep is not None and best_l_ep > 0.85 * max_epoch:
        l_offset = (-65, 20)
    else:
        l_offset = (20, 20)

    def add_headroom(ax, vals, pad_top=0.25, pad_bot=0.08):
        """Thêm khoảng đệm trên dưới cho trục Y để đường vẽ và chú thích không chạm mép."""
        if len(vals) == 0: return
        min_v, max_v = float(np.min(vals)), float(np.max(vals))
        span = (max_v - min_v) if max_v != min_v else (abs(max_v) * 0.1 or 1.0)
        ax.set_ylim(min_v - span * pad_bot, max_v + span * pad_top)

    # --- Trục 1 (Bên trái): PSNR (dB) ---
    ax1.set_xlabel("Epochs", fontsize=12, fontweight='bold', labelpad=8)
    if len(val_psnr) > 0:
        ax1.set_ylabel("PSNR (dB) ↑", color=c_psnr, fontsize=12, fontweight='bold')
        line1, = ax1.plot(ep_psnr, val_psnr, color=c_psnr, linewidth=2.2, label="PSNR")
        ax1.tick_params(axis='y', labelcolor=c_psnr, labelsize=10)
        add_headroom(ax1, val_psnr, pad_top=0.25, pad_bot=0.08)
        lines.append(line1)
        labels.append(f"PSNR (Best: {best_p_val:.2f} dB @ Ep {best_p_ep})")

        # Đánh dấu sao tại điểm Max PSNR
        ax1.scatter([best_p_ep], [best_p_val], color=c_psnr, s=150, marker='*', zorder=6,
                    edgecolors='black', linewidths=0.6)
        ax1.annotate(f"★ {best_p_val:.2f} dB (Ep {best_p_ep})",
                    xy=(best_p_ep, best_p_val), xytext=p_offset,
                    textcoords="offset points", fontsize=9, fontweight='bold', color=c_psnr,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c_psnr, lw=1.1, alpha=0.95),
                    arrowprops=dict(arrowstyle="->", color=c_psnr, lw=1.1, shrinkA=2, shrinkB=4))

    ax1.grid(True, linestyle=":", alpha=0.5)

    # --- Trục 2 (Bên phải 1): SSIM ---
    if len(val_ssim) > 0:
        ax2 = ax1.twinx()
        ax2.set_ylabel("SSIM ↑", color=c_ssim, fontsize=12, fontweight='bold')
        line2, = ax2.plot(ep_ssim, val_ssim, color=c_ssim, linewidth=2.0, linestyle="--", label="SSIM")
        ax2.tick_params(axis='y', labelcolor=c_ssim, labelsize=10)
        add_headroom(ax2, val_ssim, pad_top=0.25, pad_bot=0.08)
        lines.append(line2)
        labels.append(f"SSIM (Best: {best_s_val:.4f} @ Ep {best_s_ep})")

        # Đánh dấu sao tại điểm Max SSIM
        ax2.scatter([best_s_ep], [best_s_val], color=c_ssim, s=150, marker='*', zorder=6,
                    edgecolors='black', linewidths=0.6)
        ax2.annotate(f"★ {best_s_val:.4f} (Ep {best_s_ep})",
                    xy=(best_s_ep, best_s_val), xytext=s_offset,
                    textcoords="offset points", fontsize=9, fontweight='bold', color=c_ssim,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c_ssim, lw=1.1, alpha=0.95),
                    arrowprops=dict(arrowstyle="->", color=c_ssim, lw=1.1, shrinkA=2, shrinkB=4))

    # --- Trục 3 (Bên phải 2 - Dịch ra ngoài 65px): LPIPS ---
    if len(val_lpips) > 0:
        ax3 = ax1.twinx()
        ax3.spines['right'].set_position(('outward', 65))
        ax3.set_ylabel("LPIPS ↓", color=c_lpips, fontsize=12, fontweight='bold')
        line3, = ax3.plot(ep_lpips, val_lpips, color=c_lpips, linewidth=2.0, linestyle="-.", label="LPIPS")
        ax3.tick_params(axis='y', labelcolor=c_lpips, labelsize=10)
        add_headroom(ax3, val_lpips, pad_top=0.25, pad_bot=0.08)
        lines.append(line3)
        labels.append(f"LPIPS (Best: {best_l_val:.4f} @ Ep {best_l_ep})")

        # Đánh dấu sao tại điểm Min LPIPS
        ax3.scatter([best_l_ep], [best_l_val], color=c_lpips, s=150, marker='*', zorder=6,
                    edgecolors='black', linewidths=0.6)
        ax3.annotate(f"★ {best_l_val:.4f} (Ep {best_l_ep})",
                    xy=(best_l_ep, best_l_val), xytext=l_offset,
                    textcoords="offset points", fontsize=9, fontweight='bold', color=c_lpips,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=c_lpips, lw=1.1, alpha=0.95),
                    arrowprops=dict(arrowstyle="->", color=c_lpips, lw=1.1, shrinkA=2, shrinkB=4))

    # TẬP TRUNG KHU VỰC TRÊN CÙNG: DÀNH RIÊNG CHO TITLE VÀ LEGEND (RECT GIỚI HẠN VÙNG VẼ ĐỒ THỊ)
    fig.tight_layout(rect=[0, 0, 0.94, 0.88])

    fig.suptitle("Validation Curves: PSNR, SSIM & LPIPS", fontsize=14, fontweight='bold', y=0.98)
    fig.legend(handles=lines, labels=labels, loc="upper center", bbox_to_anchor=(0.48, 0.935),
               ncol=3, frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=10,
               borderpad=0.5, columnspacing=2.0, handlelength=2.5)

    # LƯU DƯỚI DẠNG JPG (Đảm bảo nền trắng)
    chart_path = os.path.join(save_dir, f"{prefix}.jpg")
    fig.savefig(chart_path, format="jpg", dpi=200, facecolor='white', pil_kwargs={'quality': 95})
    plt.close(fig)
    print(f"[Learning Curve Chart Saved]: {chart_path}")

    # -------------------------------------------------------------------------
    # NẾU CÓ DỮ LIỆU LOSS: LƯU THÊM BIỂU ĐỒ LOSSES DẠNG JPG
    # -------------------------------------------------------------------------
    _, loss_vals = extract_series(data, "total_loss")
    if len(loss_vals) > 0:
        fig_loss, ax_loss = plt.subplots(figsize=(10, 5.5), dpi=200)
        ep_loss, val_loss = extract_series(data, "total_loss")
        ax_loss.plot(ep_loss, val_loss, color="#d62728", linewidth=2.2, label="Total Loss")
        
        min_idx = np.argmin(val_loss)
        ax_loss.scatter([ep_loss[min_idx]], [val_loss[min_idx]], color="#d62728", s=60, zorder=5)
        ax_loss.annotate(f"Min Loss: {val_loss[min_idx]:.4f} (Ep {ep_loss[min_idx]})",
                        xy=(ep_loss[min_idx], val_loss[min_idx]), xytext=(-30, 15),
                        textcoords="offset points", fontsize=9, fontweight='bold', color="#d62728",
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728", lw=1.2, alpha=0.85))
        
        # Vẽ thêm các loss thành phần nếu có
        comp_configs = [
            ("l1", "L1 Loss", "#1f77b4", "--"),
            ("d_ssim", "D(SSIM) Loss", "#ff7f0e", "--"),
            ("p_vgg", "P(VGG) Loss", "#2ca02c", "--"),
            ("edge", "Edge Loss", "#9467bd", ":"),
        ]
        for k, name, c, ls in comp_configs:
            ep_c, val_c = extract_series(data, k)
            if len(val_c) > 0 and np.any(val_c > 0):
                ax_loss.plot(ep_c, val_c, label=name, color=c, linestyle=ls, linewidth=1.5, alpha=0.8)

        ax_loss.set_title("Training Loss Curve", fontsize=13, fontweight='bold', pad=10)
        ax_loss.set_xlabel("Epochs", fontsize=11, fontweight='bold')
        ax_loss.set_ylabel("Loss Value", fontsize=11, fontweight='bold')
        ax_loss.grid(True, linestyle=":", alpha=0.6)
        ax_loss.legend(frameon=True, facecolor="white", edgecolor="#ddd")
        fig_loss.tight_layout()

        losses_path = os.path.join(save_dir, f"{prefix}_losses.jpg")
        fig_loss.savefig(losses_path, format="jpg", dpi=200, bbox_inches='tight', facecolor='white', pil_kwargs={'quality': 95})
        plt.close(fig_loss)
        print(f"[Losses Chart Saved]: {losses_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Parse markdown training log files (.md) and plot 3 metrics on 1 single chart (JPG format)."
    )
    parser.add_argument(
        "--logs", 
        nargs="+", 
        default=None,
        help="Path to one or multiple log files (.md) or glob patterns (*.md)"
    )
    parser.add_argument(
        "--dir", 
        type=str, 
        default=None,
        help="Directory containing metrics*.md files to merge"
    )
    parser.add_argument(
        "--out", 
        type=str, 
        default=None,
        help="Output directory to save plots (default: directory of the first log file)"
    )
    args = parser.parse_args()

    # File log mẫu mặc định nếu không truyền qua CLI
    default_log = r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\weights_and_results\LoLv1\Mamba_SL_IG_EdgeLCA\Denoise\metrics2026-09-16-095700.md"

    # Xác định danh sách nguồn file
    if args.dir:
        input_sources = args.dir
    elif args.logs:
        input_sources = args.logs
    else:
        # =========================================================================
        # BẠN CÓ THỂ ĐIỀN DANH SÁCH FILE TRỰC TIẾP Ở ĐÂY HOẶC DÙNG DÒNG LỆNH:
        # Ví dụ 1 file:
        # input_sources = default_log
        #
        # Ví dụ nhiều file chia nhỏ:
        # input_sources = [
        #     r"weights_and_results/LoLv1/wo-Both-Two/metrics2026-09-30-070814.md",
        #     r"weights_and_results/LoLv1/wo-Both-Two/metrics2026-09-30-085202.md",
        # ]
        # =========================================================================
        input_sources = default_log

    print("\n[1/3] Parsing and merging log data (excluding GT Mean)...")
    data, metadata, valid_files = parse_log_files(input_sources)
    
    # Xác định thư mục lưu ảnh biểu đồ
    if args.out:
        out_dir = args.out
    else:
        out_dir = os.path.dirname(os.path.abspath(valid_files[0]))

    print(f"[2/3] Rendering and saving Learning Curve (.jpg) to: {out_dir}")
    plot_learning_curves(data, save_dir=out_dir)

    print(f"[3/3] Training statistics summary:")
    print_summary_table(data, metadata, valid_files)


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import os

# Đường dẫn tới file log
log_file = r"E:\PythonFile\Project\IGMamba-EdgeLCA-LLIE\weights_and_results\LoLv1\wo-FGLoss\metrics2026-10-01-021716.md"
out_dir = os.path.dirname(log_file)

# Khởi tạo các mảng dữ liệu
epochs, total_loss = [], []
l1, d_ssim, p_vgg, edge = [], [], [], []
psnr, ssim, lpips = [], [], []

# Đọc và parse file markdown
with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        # Bỏ qua các dòng không phải là dữ liệu bảng
        if not line.startswith("|") or "Epochs" in line or "---" in line:
            continue
            
        cols = [c.strip() for c in line.split("|")]
        if len(cols) < 14:
            continue
            
        # Chỉ lấy dữ liệu của dòng Note "-" (giá trị bình thường, bỏ qua GT Mean)
        note = cols[13]
        if note == "-":
            epochs.append(int(cols[1]))
            total_loss.append(float(cols[2]))
            l1.append(float(cols[3]))
            d_ssim.append(float(cols[5]))
            p_vgg.append(float(cols[6]))
            edge.append(float(cols[7]))
            psnr.append(float(cols[10]))
            ssim.append(float(cols[11]))
            lpips.append(float(cols[12]))

# ========================================================
# Biểu đồ 1: Learning Curve (Nhiều trục Y)
# ========================================================
fig, ax1 = plt.subplots(figsize=(10, 6))

color1 = 'tab:red'
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Total Loss / LPIPS', color=color1)
line1, = ax1.plot(epochs, total_loss, color=color1, label='Total Loss', marker='o', markersize=4)
line2, = ax1.plot(epochs, lpips, color='tab:orange', label='LPIPS', marker='s', markersize=4)
ax1.tick_params(axis='y', labelcolor=color1)

ax2 = ax1.twinx()
color2 = 'tab:blue'
ax2.set_ylabel('PSNR', color=color2)
line3, = ax2.plot(epochs, psnr, color=color2, label='PSNR', marker='^', markersize=4)
ax2.tick_params(axis='y', labelcolor=color2)

ax3 = ax1.twinx()
ax3.spines['right'].set_position(('outward', 60))
color3 = 'tab:green'
ax3.set_ylabel('SSIM', color=color3)
line4, = ax3.plot(epochs, ssim, color=color3, label='SSIM', marker='d', markersize=4)
ax3.tick_params(axis='y', labelcolor=color3)

fig.tight_layout()
fig.legend(handles=[line1, line2, line3, line4], loc='upper right', bbox_to_anchor=(0.95, 0.95))
plt.grid(True, alpha=0.3)
plt.savefig(os.path.join(out_dir, 'chart1_no_title.png'), dpi=150)
plt.close()

# ========================================================
# Biểu đồ 2: Individual Losses
# ========================================================
plt.figure(figsize=(10, 6))
plt.plot(epochs, l1, label='L1 Loss', marker='o', markersize=4)
plt.plot(epochs, d_ssim, label='D(SSIM) Loss', marker='s', markersize=4)
plt.plot(epochs, p_vgg, label='P(VGG) Loss', marker='^', markersize=4)
plt.plot(epochs, edge, label='Edge Loss', marker='d', markersize=4)
plt.plot(epochs, total_loss, label='Total Loss', marker='x', markersize=4, linestyle='--')

plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(out_dir, 'chart2_no_title.png'), dpi=150)
plt.close()

print(f"Charts saved to:\n- {os.path.join(out_dir, 'chart1_no_title.png')}\n- {os.path.join(out_dir, 'chart2_no_title.png')}")

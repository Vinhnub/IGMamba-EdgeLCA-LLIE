import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
from einops import rearrange

# Them duong dan thu muc hien tai vao sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from net.CIDNet_Mamba_separable_learning import CIDNet


class IGMambaVisualizer:
    """
    Trich xuat va truc quan hoa ban do dieu bien delta_mod va c_mod
    tu tat ca cac tang IG_Mamba trong CIDNet.
    """
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.records = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """Dang ky hook de bat delta_mod va c_mod khi forward pass chay."""
        for i in range(1, 7):
            layer_attr = f"IG_Mamba_{i}"
            if not hasattr(self.model, layer_attr):
                continue
            
            mamba_module = getattr(self.model, layer_attr)
            attn_module = mamba_module.attn
            self.records[layer_attr] = {}

            def make_attn_hook(lname):
                def hook(module, inp, out):
                    visible = inp[0]
                    B, C, H_in, W_in = visible.shape
                    w = module.window_size
                    pad_h = (w - H_in % w) % w
                    pad_w = (w - W_in % w) % w
                    H = H_in + pad_h
                    W = W_in + pad_w
                    self.records[lname]['H'] = H
                    self.records[lname]['W'] = W
                    self.records[lname]['orig_HW'] = (H_in, W_in)
                    self.records[lname]['dark_focus'] = module.dark_focus
                return hook

            def make_delta_hook(lname):
                def hook(module, inp, out):
                    self.records[lname]['raw_delta'] = out.detach().cpu()
                return hook

            def make_c_hook(lname):
                def hook(module, inp, out):
                    self.records[lname]['raw_c'] = out.detach().cpu()
                return hook

            h1 = attn_module.register_forward_hook(make_attn_hook(layer_attr))
            h2 = attn_module.i_delta_mod.register_forward_hook(make_delta_hook(layer_attr))
            h3 = attn_module.i_c_mod.register_forward_hook(make_c_hook(layer_attr))
            self.hooks.extend([h1, h2, h3])

    def unscan_to_2d(self, scans, H, W, orig_HW=None):
        """
        Khoi phuc 4 chuoi quet 1D tro lai khong gian anh 2D [B, C, H, W].
        scans: [B, 4, C, L] voi L = H * W
        """
        B, K, C, L = scans.shape
        s0 = scans[:, 0].view(B, C, H, W)
        s1 = scans[:, 1].view(B, C, H, W).flip(-1).flip(-2)
        s2 = scans[:, 2].view(B, C, W, H).transpose(-1, -2)
        s3 = scans[:, 3].view(B, C, W, H).flip(-1).flip(-2).transpose(-1, -2)
        
        # Lay trung binh cong 4 huong quet de dai dien nhat quan tren 2D
        merged = (s0 + s1 + s2 + s3) / 4.0
        
        if orig_HW is not None:
            orig_h, orig_w = orig_HW
            merged = merged[:, :, :orig_h, :orig_w]
        return merged

    def forward_and_extract(self, img_tensor):
        """
        Thuc hien forward qua model va giai ma cac ban do delta_mod, c_mod.
        img_tensor: [1, 3, H, W]
        """
        img_tensor = img_tensor.to(self.device)
        with torch.no_grad():
            output_rgb = self.model(img_tensor)

        results = {}
        for lname, data in self.records.items():
            if 'raw_delta' not in data or 'raw_c' not in data:
                continue

            H, W = data['H'], data['W']
            orig_HW = data['orig_HW']
            L = H * W
            B = img_tensor.shape[0]

            # Reshape lai chuoi quet 4 huong: [B, 4, dim, L]
            delta = rearrange(data['raw_delta'], '(b k l) r -> b k r l', b=B, k=4, l=L)
            c = rearrange(data['raw_c'], '(b k l) s -> b k s l', b=B, k=4, l=L)

            if data['dark_focus']:
                delta = -delta
                c = -c

            # Khoi phuc ve 2D: [B, Channels, H, W]
            delta_2d = self.unscan_to_2d(delta, H, W, orig_HW)
            c_2d = self.unscan_to_2d(c, H, W, orig_HW)

            # Tong hop theo chieu kenh (Mean across channels) de tao anh 2D
            delta_map = delta_2d.mean(dim=1)[0].numpy()
            c_map = c_2d.mean(dim=1)[0].numpy()

            # Do lon dieu bien (L2 Norm / Magnitude)
            delta_mag = torch.norm(delta_2d, dim=1)[0].numpy()
            c_mag = torch.norm(c_2d, dim=1)[0].numpy()

            results[lname] = {
                'delta_map': delta_map,
                'c_map': c_map,
                'delta_mag': delta_mag,
                'c_mag': c_mag,
                'orig_HW': orig_HW,
                'dt_rank': delta.shape[2],
                'd_state': c.shape[2],
            }

        return output_rgb, results

    def remove_hooks(self):
        """Huy toan bo hook khi hoan tat."""
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


def visualize_and_save(input_img_pil, output_img_tensor, results, save_path="results/ig_mamba_maps/ig_mamba_overview.png", save_individual_dir=None):
    """
    Ve anh tong hop Grid gom anh goc, anh tang cuong va 6 tang IG_Mamba (delta_mod va c_mod).
    """
    layer_names = list(results.keys())
    num_layers = len(layer_names)
    
    layer_desc = {
        'IG_Mamba_1': 'Layer 1 (Enc 1 - Scale 1/2)',
        'IG_Mamba_2': 'Layer 2 (Enc 2 - Scale 1/4)',
        'IG_Mamba_3': 'Layer 3 (Bottleneck Enc - Scale 1/8)',
        'IG_Mamba_4': 'Layer 4 (Bottleneck Dec - Scale 1/8)',
        'IG_Mamba_5': 'Layer 5 (Dec 2 - Scale 1/4)',
        'IG_Mamba_6': 'Layer 6 (Dec 1 - Scale 1/2)',
    }

    output_np = output_img_tensor[0].detach().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    input_np = np.array(input_img_pil) / 255.0

    fig = plt.figure(figsize=(20, 3.2 * (num_layers + 1)))
    gs = fig.add_gridspec(num_layers + 1, 4, width_ratios=[1, 1, 1, 1])

    # 1. Ve Input va Output
    ax_in = fig.add_subplot(gs[0, 0:2])
    ax_in.imshow(input_np)
    ax_in.set_title("Input (Low-Light Image)", fontsize=13, fontweight='bold')
    ax_in.axis('off')

    ax_out = fig.add_subplot(gs[0, 2:4])
    ax_out.imshow(output_np)
    ax_out.set_title("Enhanced Output (CIDNet)", fontsize=13, fontweight='bold')
    ax_out.axis('off')

    # 2. Ve delta_mod va c_mod cho tung tang
    for idx, lname in enumerate(layer_names):
        row = idx + 1
        data = results[lname]
        d_map = data['delta_map']
        c_map = data['c_map']
        title_prefix = layer_desc.get(lname, lname)
        h, w = data['orig_HW']

        # Delta Mod (Mean)
        ax_delta = fig.add_subplot(gs[row, 0:2])
        vmax_d = max(abs(float(d_map.min())), abs(float(d_map.max())), 1e-4)
        im_d = ax_delta.imshow(d_map, cmap='coolwarm', vmin=-vmax_d, vmax=vmax_d)
        ax_delta.set_title(f"{title_prefix}\nDelta-Modulation (delta_mod) | Res: {w}x{h} | dt_rank: {data['dt_rank']}", fontsize=11)
        ax_delta.axis('off')
        cbar_d = plt.colorbar(im_d, ax=ax_delta, fraction=0.03, pad=0.04)
        cbar_d.ax.tick_params(labelsize=8)

        # C Mod (Mean)
        ax_c = fig.add_subplot(gs[row, 2:4])
        im_c = ax_c.imshow(c_map, cmap='viridis')
        ax_c.set_title(f"{title_prefix}\nC-Modulation (c_mod) | Res: {w}x{h} | d_state: {data['d_state']}", fontsize=11)
        ax_c.axis('off')
        cbar_c = plt.colorbar(im_c, ax=ax_c, fraction=0.03, pad=0.04)
        cbar_c.ax.tick_params(labelsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[THANH CONG] Da luu anh truc quan hoa tong hop tai: {save_path}")

    # Neu co yeu cau luu rieng le tung anh
    if save_individual_dir:
        os.makedirs(save_individual_dir, exist_ok=True)
        for lname, data in results.items():
            # Luu delta
            plt.figure(figsize=(6, 5))
            vmax_d = max(abs(float(data['delta_map'].min())), abs(float(data['delta_map'].max())), 1e-4)
            plt.imshow(data['delta_map'], cmap='coolwarm', vmin=-vmax_d, vmax=vmax_d)
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"{lname} - delta_mod")
            plt.axis('off')
            plt.savefig(os.path.join(save_individual_dir, f"{lname}_delta_mod.png"), dpi=150, bbox_inches='tight')
            plt.close()

            # Luu c
            plt.figure(figsize=(6, 5))
            plt.imshow(data['c_map'], cmap='viridis')
            plt.colorbar(fraction=0.046, pad=0.04)
            plt.title(f"{lname} - c_mod")
            plt.axis('off')
            plt.savefig(os.path.join(save_individual_dir, f"{lname}_c_mod.png"), dpi=150, bbox_inches='tight')
            plt.close()
        print(f"[THANH CONG] Da luu cac anh rieng le tai thu muc: {save_individual_dir}")


def main():
    parser = argparse.ArgumentParser(description="In ban do delta_mod va c_mod trong cac tang IG_Mamba duoi dang anh")
    parser.add_argument("--image_path", type=str, default="", help="Duong dan den anh dau vao (PNG/JPG)")
    parser.add_argument("--weights", type=str, default="", help="Duong dan den file weights .pth (tuy chon)")
    parser.add_argument("--output", type=str, default="results/ig_mamba_maps/ig_mamba_overview.png", help="Duong dan luu anh tong hop")
    parser.add_argument("--save_individual", action="store_true", help="Luu rieng le tung heatmap cho moi tang")
    parser.add_argument("--img_size", type=int, default=256, help="Kich thuoc resize anh (mac dinh 256 de chay nhanh tren CPU/Windows). Dat 0 de giu nguyen anh goc.")
    parser.add_argument("--channels", nargs=4, type=int, default=[36, 36, 72, 144], help="Danh sach channel cau hinh model")
    parser.add_argument("--heads", nargs=4, type=int, default=[1, 2, 4, 8], help="Danh sach heads cau hinh model")
    args = parser.parse_args()

    # 1. Tim anh dau vao
    img_path = args.image_path
    if not img_path or not os.path.exists(img_path):
        candidates = [
            "mydata/dataset/dataset/LOLv1/test/low/1.png",
            "mydata/dataset/dataset/LOLv1/test/low/22.png",
            "mydata/dataset/LOLv1/test/low/1.png"
        ]
        for c in candidates:
            if os.path.exists(c):
                img_path = c
                break
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Su dung thiet bi: {device}")

    # Load anh
    if img_path and os.path.exists(img_path):
        print(f"Doc anh dau vao tu: {img_path}")
        pil_img = Image.open(img_path).convert('RGB')
    else:
        print("Khong tim thay anh thuc te, tu dong sinh anh mau dummy...")
        dummy_arr = np.random.randint(20, 80, (256, 256, 3), dtype=np.uint8)
        pil_img = Image.fromarray(dummy_arr)

    # Resize neu can
    if args.img_size > 0:
        pil_img = pil_img.resize((args.img_size, args.img_size), Image.BILINEAR)
        print(f"Anh duoc resize ve kich thuoc: {args.img_size}x{args.img_size}")

    # Tien xu ly anh thanh Tensor
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(device)

    # 2. Khoi tao mo hinh
    print(f"Khoi tao CIDNet voi channels={args.channels}, heads={args.heads}...")
    model = CIDNet(channels=args.channels, heads=args.heads).to(device)

    if args.weights and os.path.exists(args.weights):
        print(f"Tai trong so tu: {args.weights}...")
        ckpt = torch.load(args.weights, map_location=device)
        state_dict = ckpt['model'] if 'model' in ckpt else ckpt
        model_dict = model.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        model.load_state_dict(model_dict, strict=False)
        print(f"Da nap thanh cong {len(matched)}/{len(model_dict)} keys.")
    else:
        print("Chay voi trong so khoi tao cua mo hinh.")

    # 3. Trich xuat va truc quan hoa
    visualizer = IGMambaVisualizer(model, device=device)
    print("Dang chay forward pass va trich xuat delta_mod & c_mod tu 6 tang IG_Mamba...")
    out_rgb, results = visualizer.forward_and_extract(img_tensor)

    # In thong so thong ke ra man hinh console
    print("\n" + "="*75)
    print(f"{'Tang':<15} | {'Do phan giai':<14} | {'Delta-Mod Min / Max':<22} | {'C-Mod Min / Max':<22}")
    print("="*75)
    for lname, data in results.items():
        d_min, d_max = data['delta_map'].min(), data['delta_map'].max()
        c_min, c_max = data['c_map'].min(), data['c_map'].max()
        h, w = data['orig_HW']
        print(f"{lname:<15} | {w}x{h:<11} | {d_min:+.4f} / {d_max:+.4f}{'':<6} | {c_min:+.4f} / {c_max:+.4f}")
    print("="*75)

    # 4. Xuat anh
    indiv_dir = os.path.dirname(args.output) if args.save_individual else None
    visualize_and_save(
        input_img_pil=pil_img,
        output_img_tensor=out_rgb,
        results=results,
        save_path=args.output,
        save_individual_dir=indiv_dir
    )

    visualizer.remove_hooks()


if __name__ == "__main__":
    main()

"""
analyze_mamba_maps.py
=====================
Load BẤT KỲ checkpoint nào của project (IG+EdgeLCA, wo-EdgeLCA, wo-IGMamba, wo-Three, ...)
và xuất các bản đồ của khối Mamba:

    - Delta base   : softplus(dt_proj(dts_base) + bias)                (chỉ từ nhánh HV)
    - Delta mode   : softplus(dt_proj(dts_base ± i_delta) + bias)      (có dẫn hướng I)
    - Delta-Delta  : Delta mode - Delta base                           (phần dịch chuyển do nhánh I)
    - C base       : ||C_base||   (norm theo d_state)
    - C mode       : ||C_base ± i_c||
    - C-Delta      : ||C mode|| - ||C base||                            (CÓ phụ thuộc dấu)
    - C mod mean   : mean_s(± i_c)                                       (CÓ phụ thuộc dấu)
  (Với Standard Mamba / wo-IGMamba: chỉ có Delta và ||C||, Delta-Delta = N/A.)

Kiến trúc được TỰ ĐỘNG nhận diện từ key của state_dict:
    IG_Mamba_*      + edge_extractor -> ig_edge     (net/CIDNet_Mamba_separable_learning_edge.py)
    IG_Mamba_*      (không edge)     -> ig_noedge   (net/CIDNet_Mamba_separable_learning.py)
    StandardMamba_* + edge_extractor -> std_edge    (wo-IGMamba)
    StandardMamba_* (không edge)     -> std_noedge  (wo-Three / wo-Both-Two)

Cờ focus (dark_focus KHÔNG được lưu trong checkpoint!):
    --focus dark | light | auto   (auto = đọc 'dark_focus:' trong metrics*.md cạnh checkpoint)

Kiểm tra tương đương chính xác (--equiv_test):
    A: focus F,      trọng số gốc
    B: focus not F,  đảo dấu weight/bias của i_delta_mod[0] và i_c_mod[0]   -> phải TRÙNG A
    C: focus not F,  trọng số gốc (chỉ lật cờ)                               -> lệch so với A

Ví dụ:
    python analyze_mamba_maps.py --weights weights_and_results/LoLv1/wo-FGLoss/train/epoch_485.pth
    python analyze_mamba_maps.py --weights a.pth b.pth --focus light --layers 1 6
    python analyze_mamba_maps.py --weights weights_and_results/LoLv1/wo-FGLoss/train/epoch_485.pth --equiv_test
"""

import os
import re
import sys
import glob
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from einops import rearrange

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

try:
    from scipy.stats import spearmanr, pearsonr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from net.CIDNet_Mamba_separable_learning_edge import CIDNet as CIDNetEdge
from net.CIDNet_Mamba_separable_learning import CIDNet as CIDNetNoEdge
from net.IG_Mamba import IG_Mamba, IG_Attention
from net.standard_mamba import StandardMamba


# =============================================================================
# 1. CÁC BIẾN THỂ KIẾN TRÚC
# =============================================================================
class CIDNet_StdMamba_Edge(CIDNetEdge):
    """wo-IGMamba: EdgeLCA + Standard Mamba (giống run_standard_mamba_00760.py)."""

    def __init__(self, channels=[36, 36, 72, 144], heads=[1, 2, 4, 8], norm=False, **kwargs):
        super().__init__(channels=channels, heads=heads, norm=norm)
        _, ch2, ch3, ch4 = channels
        for i in range(1, 7):
            if hasattr(self, f'IG_Mamba_{i}'):
                delattr(self, f'IG_Mamba_{i}')
        self.StandardMamba_1 = StandardMamba(ch2)
        self.StandardMamba_2 = StandardMamba(ch3)
        self.StandardMamba_3 = StandardMamba(ch4)
        self.StandardMamba_4 = StandardMamba(ch4)
        self.StandardMamba_5 = StandardMamba(ch3)
        self.StandardMamba_6 = StandardMamba(ch2)

    @property
    def dark_focus(self):
        return None

    @dark_focus.setter
    def dark_focus(self, val):
        pass

    def forward(self, x, return_feats=False):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2:3, :, :].to(dtypes)
        edge_map = self.edge_extractor(i).to(dtypes)
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0, hv_jump0 = i_enc0, hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1, edge_map=edge_map)
        hv_2 = self.StandardMamba_1(hv_1)
        v_jump1, hv_jump1 = i_enc2, hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2, edge_map=edge_map)
        hv_3 = self.StandardMamba_2(hv_2)
        v_jump2, hv_jump2 = i_enc3, hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3, edge_map=edge_map)
        hv_4 = self.StandardMamba_3(hv_3)
        i_dec4 = self.I_LCA4(i_enc4, hv_4, edge_map=edge_map)
        hv_4 = self.StandardMamba_4(hv_4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3, edge_map=edge_map)
        hv_2 = self.StandardMamba_5(hv_3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        i_dec1 = self.I_LCA6(i_dec2, hv_2, edge_map=edge_map)
        hv_1 = self.StandardMamba_6(hv_2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        return self.trans.PHVIT(output_hvi)


class CIDNet_StdMamba_NoEdge(CIDNetNoEdge):
    """wo-Three / wo-Both-Two: I_LCA thường + Standard Mamba (không EdgeLCA, không IG)."""

    def __init__(self, channels=[36, 36, 72, 144], heads=[1, 2, 4, 8], norm=False, **kwargs):
        super().__init__(channels=channels, heads=heads, norm=norm)
        _, ch2, ch3, ch4 = channels
        for i in range(1, 7):
            if hasattr(self, f'IG_Mamba_{i}'):
                delattr(self, f'IG_Mamba_{i}')
        self.StandardMamba_1 = StandardMamba(ch2)
        self.StandardMamba_2 = StandardMamba(ch3)
        self.StandardMamba_3 = StandardMamba(ch4)
        self.StandardMamba_4 = StandardMamba(ch4)
        self.StandardMamba_5 = StandardMamba(ch3)
        self.StandardMamba_6 = StandardMamba(ch2)

    @property
    def dark_focus(self):
        return None

    @dark_focus.setter
    def dark_focus(self, val):
        pass

    def forward(self, x, return_feats=False):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)
        i_jump0, hv_jump0 = i_enc0, hv_0

        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.StandardMamba_1(hv_1)
        v_jump1, hv_jump1 = i_enc2, hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.StandardMamba_2(hv_2)
        v_jump2, hv_jump2 = i_enc3, hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.StandardMamba_3(hv_3)
        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.StandardMamba_4(hv_4)

        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)
        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.StandardMamba_5(hv_3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.StandardMamba_6(hv_2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        return self.trans.PHVIT(output_hvi)


ARCH_BUILDERS = {
    'ig_edge': lambda df: CIDNetEdge(dark_focus=df),
    'ig_noedge': lambda df: CIDNetNoEdge(dark_focus=df),
    'std_edge': lambda df: CIDNet_StdMamba_Edge(),
    'std_noedge': lambda df: CIDNet_StdMamba_NoEdge(),
}


def load_state_dict_any(path):
    try:
        ckpt = torch.load(path, map_location='cpu', weights_only=True)
    except Exception:
        ckpt = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'params', 'net'):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    # bỏ prefix 'module.' (DDP)
    return {(k[7:] if k.startswith('module.') else k): v for k, v in ckpt.items()}


def detect_arch(state):
    has_ig = any(k.startswith('IG_Mamba_') for k in state)
    has_std = any(k.startswith('StandardMamba_') for k in state)
    has_edge = any(k.startswith('edge_extractor') for k in state)
    if has_ig:
        return 'ig_edge' if has_edge else 'ig_noedge'
    if has_std:
        return 'std_edge' if has_edge else 'std_noedge'
    raise RuntimeError("Không nhận diện được kiến trúc (không thấy IG_Mamba_* hay StandardMamba_*).")


def find_focus_from_metrics(ckpt_path, max_up=3):
    """Tìm 'dark_focus: True/False' trong metrics*.md ở thư mục checkpoint và các thư mục cha."""
    d = os.path.dirname(os.path.abspath(ckpt_path))
    for _ in range(max_up + 1):
        values = set()
        for md in glob.glob(os.path.join(d, 'metrics*.md')):
            try:
                with open(md, 'r', encoding='utf-8', errors='ignore') as f:
                    for line in f.readlines()[:40]:
                        m = re.match(r'\s*dark_focus:\s*(True|False)', line)
                        if m:
                            values.add(m.group(1) == 'True')
            except OSError:
                pass
        if values:
            if len(values) > 1:
                return None, f"metrics*.md trong {d} có giá trị dark_focus mâu thuẫn"
            return values.pop(), f"đọc từ metrics*.md trong {d}"
        d = os.path.dirname(d)
    return None, "không tìm thấy dòng 'dark_focus:' trong metrics*.md"


def build_model(ckpt_path, dark_focus, device, arch='auto'):
    state = load_state_dict_any(ckpt_path)
    if arch == 'auto':
        arch = detect_arch(state)
    model = ARCH_BUILDERS[arch](dark_focus)
    model.load_state_dict(state, strict=False)
    if arch.startswith('ig'):
        model.dark_focus = dark_focus  # đồng bộ tất cả IG_Mamba
    return model.to(device).eval(), arch


# =============================================================================
# 2. TRÍCH XUẤT BẢN ĐỒ (hook trên khối attention)
# =============================================================================
def unscan_mean(scans, H, W, crop_hw):
    """scans: [B, 4, C, L] -> [B, C, h, w] (trung bình 4 hướng quét, crop vùng hợp lệ)."""
    B, K, C, L = scans.shape
    s0 = scans[:, 0].reshape(B, C, H, W)
    s1 = scans[:, 1].reshape(B, C, H, W).flip(-1).flip(-2)
    s2 = scans[:, 2].reshape(B, C, W, H).transpose(-1, -2)
    s3 = scans[:, 3].reshape(B, C, W, H).flip(-1).flip(-2).transpose(-1, -2)
    merged = (s0 + s1 + s2 + s3) / 4.0
    h, w = crop_hw
    return merged[:, :, :h, :w]


class MambaMapExtractor:
    def __init__(self, model, layers=range(1, 7)):
        self.model = model
        self.records = {}
        self.hooks = []
        self.img_hw = None
        for i in layers:
            if hasattr(model, f'IG_Mamba_{i}'):
                name, mod, kind = f'IG_Mamba_{i}', getattr(model, f'IG_Mamba_{i}').attn, 'ig'
            elif hasattr(model, f'StandardMamba_{i}'):
                name, mod, kind = f'StandardMamba_{i}', getattr(model, f'StandardMamba_{i}').attn, 'std'
            else:
                continue
            self.hooks.append(mod.register_forward_hook(self._make_hook(i, name, mod, kind)))

    def remove(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def _crop_hw(self, h_feat, w_feat):
        # vùng hợp lệ (bỏ phần pad ảnh về bội số 8) ở scale của tầng hiện tại
        H_img, W_img, H_pad, W_pad = self.img_hw
        return (int(round(h_feat * H_img / H_pad)), int(round(w_feat * W_img / W_pad)))

    def _make_hook(self, idx, name, module, kind):
        def hook(m, inp, out):
            with torch.no_grad():
                x_in = inp[0]
                oh, ow = x_in.shape[2], x_in.shape[3]
                crop = self._crop_hw(oh, ow)
                if kind == 'ig':
                    self.records[idx] = self._ig_maps(module, inp[0], inp[1], crop, name)
                else:
                    self.records[idx] = self._std_maps(module, inp[0], crop, name)
        return hook

    @staticmethod
    def _pad(module, x):
        if module.enable_padding:
            x, _, _ = module.pad_to_window_size(x, module.window_size)
        return x

    def _ig_maps(self, mod, visible, infrared, crop, name):
        vis = self._pad(mod, visible)
        inf = self._pad(mod, infrared)
        B, _, H, W = vis.shape
        L = H * W

        x_vis, _ = mod.in_proj_hv(rearrange(vis, 'b c h w -> b h w c')).chunk(2, dim=-1)
        x_inf, _ = mod.in_proj_i(rearrange(inf, 'b c h w -> b h w c')).chunk(2, dim=-1)
        x_vis = mod.act_hv(mod.conv2d_hv(x_vis.permute(0, 3, 1, 2).contiguous()))
        x_inf = mod.act_i(mod.conv2d_i(x_inf.permute(0, 3, 1, 2).contiguous()))

        xs_hv = mod._get_four_scans(x_vis)
        xs_i = mod._get_four_scans(x_inf)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs_hv, mod.x_proj_weight)
        dts_base, _, Cs_base = torch.split(x_dbl, [mod.dt_rank, mod.d_state, mod.d_state], dim=2)

        xs_i_t = rearrange(xs_i, "b k d l -> (b k l) d")
        i_delta = rearrange(mod.i_delta_mod(xs_i_t), "(b k l) r -> b k r l", b=B, k=mod.K, l=L)
        i_c = rearrange(mod.i_c_mod(xs_i_t), "(b k l) s -> b k s l", b=B, k=mod.K, l=L)
        if mod.dark_focus:
            i_delta, i_c = -i_delta, -i_c

        dts_mode = dts_base + i_delta
        Cs_mode = Cs_base + i_c

        bias = mod.dt_projs_bias.view(1, mod.K, mod.d_inner, 1)
        d_base = F.softplus(torch.einsum("b k r l, k d r -> b k d l", dts_base, mod.dt_projs_weight) + bias)
        d_mode = F.softplus(torch.einsum("b k r l, k d r -> b k d l", dts_mode, mod.dt_projs_weight) + bias)

        d_base_2d = unscan_mean(d_base, H, W, crop).mean(1)[0]
        d_mode_2d = unscan_mean(d_mode, H, W, crop).mean(1)[0]
        c_base_2d = unscan_mean(Cs_base, H, W, crop)[0]
        c_mode_2d = unscan_mean(Cs_mode, H, W, crop)[0]
        i_c_2d = unscan_mean(i_c, H, W, crop)[0]

        c_base_norm = c_base_2d.norm(dim=0)
        c_mode_norm = c_mode_2d.norm(dim=0)
        return {
            'name': name, 'kind': 'ig', 'dark_focus': bool(mod.dark_focus),
            'delta_base': d_base_2d.cpu().numpy(),
            'delta_mode': d_mode_2d.cpu().numpy(),
            'delta_delta': (d_mode_2d - d_base_2d).cpu().numpy(),
            'c_base': c_base_norm.cpu().numpy(),
            'c_mode': c_mode_norm.cpu().numpy(),
            'c_delta': (c_mode_norm - c_base_norm).cpu().numpy(),
            'c_mod_mean': i_c_2d.mean(0).cpu().numpy(),
            'c_mod_norm': i_c_2d.norm(dim=0).cpu().numpy(),
        }

    def _std_maps(self, mod, x, crop, name):
        xp = self._pad(mod, x)
        B, _, H, W = xp.shape
        x_val, _ = mod.in_proj(rearrange(xp, 'b c h w -> b h w c')).chunk(2, dim=-1)
        x_val = mod.act(mod.conv2d(x_val.permute(0, 3, 1, 2).contiguous()))
        xs = mod._get_four_scans(x_val)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, mod.x_proj_weight)
        dts, _, Cs = torch.split(x_dbl, [mod.dt_rank, mod.d_state, mod.d_state], dim=2)
        bias = mod.dt_projs_bias.view(1, mod.K, mod.d_inner, 1)
        delta = F.softplus(torch.einsum("b k r l, k d r -> b k d l", dts, mod.dt_projs_weight) + bias)
        d2d = unscan_mean(delta, H, W, crop).mean(1)[0].cpu().numpy()
        c2d = unscan_mean(Cs, H, W, crop)[0].norm(dim=0).cpu().numpy()
        return {
            'name': name, 'kind': 'std', 'dark_focus': None,
            'delta_base': d2d, 'delta_mode': d2d, 'delta_delta': None,
            'c_base': c2d, 'c_mode': c2d, 'c_delta': None,
            'c_mod_mean': None, 'c_mod_norm': None,
        }

    @torch.no_grad()
    def run(self, img):
        """img: [1,3,H,W] trong [0,1]. Trả về (output [1,3,H,W], records)."""
        self.records = {}
        H, W = img.shape[2], img.shape[3]
        ph, pw = (8 - H % 8) % 8, (8 - W % 8) % 8
        x = F.pad(img, (0, pw, 0, ph), mode='replicate') if (ph or pw) else img
        self.img_hw = (H, W, H + ph, W + pw)
        out = self.model(x)[:, :, :H, :W]
        return out, dict(sorted(self.records.items()))


# =============================================================================
# 3. THỐNG KÊ
# =============================================================================
def luma_of(img_np):
    return 0.299 * img_np[..., 0] + 0.587 * img_np[..., 1] + 0.114 * img_np[..., 2]


def resize_map(m, hw):
    t = torch.from_numpy(np.ascontiguousarray(m)).float()[None, None]
    return F.interpolate(t, size=hw, mode='area')[0, 0].numpy()


def corr_stats(y, luma):
    y, l = y.flatten(), luma.flatten()
    if HAS_SCIPY:
        sp = float(spearmanr(y, l)[0])
        pe = float(pearsonr(y, l)[0])
    else:
        rk = lambda a: np.argsort(np.argsort(a)).astype(np.float64)
        pc = lambda a, b: float(np.corrcoef(a, b)[0, 1])
        sp, pe = pc(rk(y), rk(l)), pc(y, l)
    p30, p70 = np.percentile(l, 30), np.percentile(l, 70)
    return {
        'spearman': sp, 'pearson': pe,
        'mean_dark': float(y[l <= p30].mean()), 'mean_bright': float(y[l >= p70].mean()),
    }


def verdict(st):
    if st is None:
        return 'N/A'
    if st['spearman'] < -0.05 and st['mean_dark'] > st['mean_bright']:
        return 'DARK-FOCUS (tăng ở vùng tối)'
    if st['spearman'] > 0.05 and st['mean_dark'] < st['mean_bright']:
        return 'LIGHT-FOCUS (tăng ở vùng sáng)'
    return 'NEUTRAL'


def compute_stats(records, luma_full):
    stats = {}
    for idx, r in records.items():
        lm = resize_map(luma_full, r['delta_mode'].shape)
        s = {'luma': lm}
        for key in ('delta_base', 'delta_mode', 'delta_delta', 'c_base', 'c_mode', 'c_delta', 'c_mod_mean'):
            s[key] = corr_stats(r[key], lm) if r[key] is not None else None
        stats[idx] = s
    return stats


def psnr(a, b):
    mse = float(torch.mean((a.clamp(0, 1) - b.clamp(0, 1)) ** 2))
    return float('inf') if mse == 0 else 10 * np.log10(1.0 / mse)


# =============================================================================
# 4. VẼ HÌNH
# =============================================================================
def _imshow(ax, fig, data, title='', cmap='inferno', vmin=None, vmax=None, diverging=False):
    if data is None:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14, transform=ax.transAxes)
        if title:
            ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        return
    if diverging:
        lim = float(np.abs(data).max()) or 1e-8
        im = ax.imshow(data, cmap='RdBu_r', norm=TwoSlopeNorm(vcenter=0.0, vmin=-lim, vmax=lim))
    else:
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')
    ax.axis('off')
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.ax.tick_params(labelsize=8)


def _fmt(st):
    return '' if st is None else f"\nρ={st['spearman']:+.3f} | dark={st['mean_dark']:+.4f} bright={st['mean_bright']:+.4f}"


def save_layer_figure(tag, focus_str, idx, rec, st, img_np, out_np, save_path):
    fig, axes = plt.subplots(3, 3, figsize=(17, 12))
    _imshow(axes[0, 0], fig, img_np, 'Input')
    _imshow(axes[0, 1], fig, out_np, 'Output')
    _imshow(axes[0, 2], fig, st['luma'], f"Luma (scale tầng {idx})", cmap='gray')

    dvmin = float(min(rec['delta_base'].min(), rec['delta_mode'].min()))
    dvmax = float(max(rec['delta_base'].max(), rec['delta_mode'].max()))
    _imshow(axes[1, 0], fig, rec['delta_base'], 'Delta base' + _fmt(st['delta_base']), vmin=dvmin, vmax=dvmax)
    _imshow(axes[1, 1], fig, rec['delta_mode'], 'Delta mode (guided)' + _fmt(st['delta_mode']), vmin=dvmin, vmax=dvmax)
    _imshow(axes[1, 2], fig, rec['delta_delta'], 'Delta-Delta = mode - base' + _fmt(st['delta_delta']), diverging=True)

    cvmin = float(min(rec['c_base'].min(), rec['c_mode'].min()))
    cvmax = float(max(rec['c_base'].max(), rec['c_mode'].max()))
    _imshow(axes[2, 0], fig, rec['c_base'], '||C base||' + _fmt(st['c_base']), cmap='viridis', vmin=cvmin, vmax=cvmax)
    _imshow(axes[2, 1], fig, rec['c_mode'], '||C mode|| (guided)' + _fmt(st['c_mode']), cmap='viridis', vmin=cvmin, vmax=cvmax)
    _imshow(axes[2, 2], fig, rec['c_delta'], 'C-Delta = ||C mode|| - ||C base||' + _fmt(st['c_delta']), diverging=True)

    fig.suptitle(f"{tag} | {rec['name']} | focus = {focus_str} | Delta-Delta: {verdict(st['delta_delta'])}",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_individual(rec, out_dir, prefix):
    """
    Lưu từng heatmap riêng lẻ:
    - KHÔNG ghi tiêu đề lên ảnh (ảnh hoàn toàn sạch, phục vụ vẽ báo cáo / paper).
    - Tên tầng và loại đặc trưng được thể hiện rõ ràng trong tên file.
    """
    os.makedirs(out_dir, exist_ok=True)

    dvmin, dvmax = None, None
    if rec['delta_base'] is not None and rec['delta_mode'] is not None:
        dvmin = float(min(rec['delta_base'].min(), rec['delta_mode'].min()))
        dvmax = float(max(rec['delta_base'].max(), rec['delta_mode'].max()))

    cvmin, cvmax = None, None
    if rec['c_base'] is not None and rec['c_mode'] is not None:
        cvmin = float(min(rec['c_base'].min(), rec['c_mode'].min()))
        cvmax = float(max(rec['c_base'].max(), rec['c_mode'].max()))

    items = [
        ('delta_base', 'inferno', False, dvmin, dvmax),
        ('delta_mode', 'inferno', False, dvmin, dvmax),
        ('delta_delta', None, True, None, None),
        ('c_base', 'viridis', False, cvmin, cvmax),
        ('c_mode', 'viridis', False, cvmin, cvmax),
        ('c_delta', None, True, None, None),
        ('c_mod_mean', None, True, None, None),
        ('c_mod_norm', 'viridis', False, None, None),
    ]
    for key, cmap, div, vmin, vmax in items:
        if rec.get(key) is None:
            continue
        fig, ax = plt.subplots(figsize=(7, 5))
        # Không truyền title (title='') để ảnh sạch hoàn toàn
        _imshow(ax, fig, rec[key], title='', cmap=cmap or 'inferno', vmin=vmin, vmax=vmax, diverging=div)
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{prefix}_{key}.png")
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close(fig)


# =============================================================================
# 5. KIỂM TRA TƯƠNG ĐƯƠNG CHÍNH XÁC
# =============================================================================
def negate_guidance_weights(model):
    n = 0
    with torch.no_grad():
        for m in model.modules():
            if isinstance(m, IG_Attention):
                for seq in (m.i_delta_mod, m.i_c_mod):
                    lin = seq[0]
                    lin.weight.neg_()
                    if lin.bias is not None:
                        lin.bias.neg_()
                n += 1
    return n


def equivalence_test(ckpt, arch, dark_focus, img, img_np, gt, device, layers, out_dir, tag):
    fstr = lambda df: 'dark' if df else 'light'
    print("\n" + "=" * 100)
    print(f"KIỂM TRA TƯƠNG ĐƯƠNG CHÍNH XÁC — {tag}")
    print("=" * 100)

    variants = {}
    specs = [
        ('A', dark_focus, False, f"focus={fstr(dark_focus)}, trọng số gốc"),
        ('B', not dark_focus, True, f"focus={fstr(not dark_focus)}, ĐẢO DẤU i_delta_mod[0] & i_c_mod[0]"),
        ('C', not dark_focus, False, f"focus={fstr(not dark_focus)}, trọng số gốc (chỉ lật cờ)"),
    ]
    for key, df, neg, desc in specs:
        model, _ = build_model(ckpt, df, device, arch)
        n_neg = negate_guidance_weights(model) if neg else 0
        ext = MambaMapExtractor(model, layers)
        out, rec = ext.run(img)
        ext.remove()
        luma = luma_of(img_np)
        variants[key] = {'desc': desc, 'out': out.detach().cpu(), 'rec': rec,
                         'stats': compute_stats(rec, luma), 'n_neg': n_neg, 'dark_focus': df}
        del model
        if device == 'cuda':
            torch.cuda.empty_cache()

    A, B, C = variants['A'], variants['B'], variants['C']
    print(f"Số khối IG_Attention bị đảo dấu ở B: {B['n_neg']}")
    for k in 'ABC':
        print(f"  [{k}] {variants[k]['desc']}")

    report = {'checkpoint': ckpt, 'arch': arch, 'variants': {k: v['desc'] for k, v in variants.items()},
              'output': {}, 'layers': {}}

    def max_abs(a, b):
        return float(np.abs(a - b).max()) if a is not None and b is not None else float('nan')

    dAB = float((A['out'] - B['out']).abs().max())
    dAC = float((A['out'] - C['out']).abs().max())
    report['output'] = {'max_abs_A_B': dAB, 'max_abs_A_C': dAC,
                        'psnr_A_vs_B': psnr(A['out'], B['out']), 'psnr_A_vs_C': psnr(A['out'], C['out'])}
    print("\n--- OUTPUT ẢNH ---")
    print(f"max|A - B| = {dAB:.3e}   PSNR(A,B) = {report['output']['psnr_A_vs_B']:.2f} dB")
    print(f"max|A - C| = {dAC:.3e}   PSNR(A,C) = {report['output']['psnr_A_vs_C']:.2f} dB")
    if gt is not None:
        for k in 'ABC':
            report['output'][f'psnr_{k}_vs_GT'] = psnr(variants[k]['out'], gt)
        print("PSNR so với GT:  " + "  ".join(f"{k}={report['output'][f'psnr_{k}_vs_GT']:.2f} dB" for k in 'ABC'))

    keys = ['delta_base', 'delta_mode', 'delta_delta', 'c_base', 'c_mode', 'c_delta', 'c_mod_mean', 'c_mod_norm']
    print("\n--- BẢN ĐỒ THEO TẦNG: max|A-B| (phải ~0)  /  max|A-C| ---")
    print(f"{'Tầng':<12}" + "".join(f"{k:>24}" for k in keys))
    for idx in A['rec']:
        ra, rb, rc = A['rec'][idx], B['rec'][idx], C['rec'][idx]
        row = {k: (max_abs(ra[k], rb[k]), max_abs(ra[k], rc[k])) for k in keys}
        report['layers'][ra['name']] = {
            'max_abs_A_B': {k: v[0] for k, v in row.items()},
            'max_abs_A_C': {k: v[1] for k, v in row.items()},
            'spearman_delta_delta': {k: variants[k]['stats'][idx]['delta_delta']['spearman'] for k in 'ABC'},
            'spearman_c_delta': {k: variants[k]['stats'][idx]['c_delta']['spearman'] for k in 'ABC'},
            'verdict_delta_delta': {k: verdict(variants[k]['stats'][idx]['delta_delta']) for k in 'ABC'},
        }
        print(f"{ra['name']:<12}" + "".join(f"{v[0]:>11.2e} / {v[1]:<10.2e}" for v in row.values()))

    print("\n--- SPEARMAN(Delta-Delta, Luma) & KẾT LUẬN ---")
    print(f"{'Tầng':<12} | {'A':^40} | {'B':^40} | {'C':^40}")
    for idx in A['rec']:
        cells = []
        for k in 'ABC':
            s = variants[k]['stats'][idx]['delta_delta']
            cells.append(f"ρ={s['spearman']:+.4f} {verdict(s)}")
        print(f"{A['rec'][idx]['name']:<12} | " + " | ".join(f"{c:<40}" for c in cells))

    tol = 1e-4
    ok = dAB < tol and all(
        all(v < tol for v in L['max_abs_A_B'].values() if not np.isnan(v)) for L in report['layers'].values())
    report['equivalent'] = bool(ok)
    print("\nKẾT LUẬN: " + ("A ≡ B (tương đương chính xác, sai khác chỉ ở mức số học float)."
                           if ok else f"A và B KHÔNG trùng (ngưỡng {tol}) — cần kiểm tra lại!"))

    # Hình so sánh A / B / C cho tầng cuối được chọn
    last = list(A['rec'].keys())[-1]
    fig, axes = plt.subplots(3, 4, figsize=(22, 13))
    for r, k in enumerate('ABC'):
        v = variants[k]
        rec, st = v['rec'][last], v['stats'][last]
        _imshow(axes[r, 0], fig, v['out'][0].permute(1, 2, 0).clamp(0, 1).numpy(), f"[{k}] Output\n{v['desc']}")
        _imshow(axes[r, 1], fig, rec['delta_mode'], f"[{k}] Delta mode" + _fmt(st['delta_mode']))
        _imshow(axes[r, 2], fig, rec['delta_delta'], f"[{k}] Delta-Delta" + _fmt(st['delta_delta']), diverging=True)
        _imshow(axes[r, 3], fig, rec['c_delta'], f"[{k}] C-Delta" + _fmt(st['c_delta']), diverging=True)
    fig.suptitle(f"Equivalence test — {tag} — {A['rec'][last]['name']}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig_path = os.path.join(out_dir, f"equiv_test_{A['rec'][last]['name']}.png")
    plt.savefig(fig_path, dpi=130, bbox_inches='tight')
    plt.close(fig)

    with open(os.path.join(out_dir, 'equiv_test_report.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"[SAVED] {fig_path}")
    print(f"[SAVED] {os.path.join(out_dir, 'equiv_test_report.json')}")
    return report


# =============================================================================
# 6. MAIN
# =============================================================================
def guess_gt_path(img_path):
    cands = []
    for a, b in [(os.sep + 'Input' + os.sep, os.sep + 'GT' + os.sep), (os.sep + 'low' + os.sep, os.sep + 'high' + os.sep),
                 (os.sep + 'Low' + os.sep, os.sep + 'Normal' + os.sep)]:
        p = os.path.normpath(img_path)
        if a in p:
            cands.append(p.replace(a, b))
            cands.append(p.replace(a, b).replace('low', 'normal'))
    for c in cands:
        if os.path.isfile(c):
            return c
    return None


def tag_from_path(p):
    p = os.path.normpath(p)
    parts = p.split(os.sep)
    if 'weights_and_results' in parts:
        parts = parts[parts.index('weights_and_results') + 1:]
    else:
        parts = parts[-3:]
    parts[-1] = os.path.splitext(parts[-1])[0]
    return '_'.join(parts)


def main():
    ap = argparse.ArgumentParser(description="Delta base / Delta mode / Delta-Delta / C maps cho mọi checkpoint")
    ap.add_argument('--weights', nargs='+', required=True, help='Một hoặc nhiều file .pth')
    ap.add_argument('--image', default=r'dataset\LOL\LOLv2-real\Test\Input\00760.png')
    ap.add_argument('--gt', default=None, help='Ảnh GT (mặc định: tự đoán từ --image)')
    ap.add_argument('--focus', choices=['auto', 'dark', 'light'], default='auto',
                    help='dark => dark_focus=True; light => dark_focus=False; auto => đọc metrics*.md (fallback dark)')
    ap.add_argument('--arch', choices=['auto'] + list(ARCH_BUILDERS.keys()), default='auto')
    ap.add_argument('--layers', nargs='+', type=int, default=[6], help='Các tầng Mamba cần xuất (1..6)')
    ap.add_argument('--out', default=r'results\mamba_maps')
    ap.add_argument('--save_individual', action='store_true', help='Lưu thêm từng heatmap riêng lẻ')
    ap.add_argument('--save_npy', action='store_true', help='Lưu các bản đồ dạng .npz')
    ap.add_argument('--equiv_test', action='store_true', help='Chạy kiểm tra tương đương chính xác A/B/C')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    img_np = np.asarray(Image.open(args.image).convert('RGB')).astype(np.float32) / 255.0
    img = torch.from_numpy(img_np).permute(2, 0, 1)[None].to(args.device)
    gt_path = args.gt or guess_gt_path(args.image)
    gt = None
    if gt_path and os.path.isfile(gt_path):
        gt = torch.from_numpy(np.asarray(Image.open(gt_path).convert('RGB')).astype(np.float32) / 255.0)
        gt = gt.permute(2, 0, 1)[None]
    print(f"Image : {args.image}")
    print(f"GT    : {gt_path if gt is not None else '(không có)'}")
    print(f"Device: {args.device}")

    for ckpt in args.weights:
        tag = tag_from_path(ckpt)
        state = load_state_dict_any(ckpt)
        arch = detect_arch(state) if args.arch == 'auto' else args.arch
        del state

        if args.focus == 'auto':
            df, src = find_focus_from_metrics(ckpt)
            if df is None:
                df, src = True, src + " -> dùng mặc định dark (dark_focus=True). HÃY KIỂM TRA LẠI!"
        else:
            df, src = (args.focus == 'dark'), 'chỉ định qua --focus'
        focus_str = ('dark' if df else 'light') if arch.startswith('ig') else 'N/A (không có IG)'

        print("\n" + "#" * 100)
        print(f"Checkpoint: {ckpt}")
        print(f"Tag       : {tag}")
        print(f"Kiến trúc : {arch}")
        print(f"Focus     : {focus_str}   ({src})")
        print("#" * 100)

        out_dir = os.path.join(args.out, f"{tag}__{'dark' if df else 'light'}" if arch.startswith('ig') else tag)
        os.makedirs(out_dir, exist_ok=True)

        model, arch = build_model(ckpt, df, args.device, arch)
        ext = MambaMapExtractor(model, args.layers)
        out, records = ext.run(img)
        ext.remove()
        out_cpu = out.detach().cpu()
        out_np = out_cpu[0].permute(1, 2, 0).clamp(0, 1).numpy()
        stats = compute_stats(records, luma_of(img_np))
        Image.fromarray((out_np * 255).round().astype(np.uint8)).save(os.path.join(out_dir, 'output.png'))
        if gt is not None:
            print(f"PSNR(output, GT) = {psnr(out_cpu, gt):.2f} dB")

        print(f"\n{'Tầng':<16} {'Kích thước':<11} {'ρ(ΔΔ,L)':>9} {'ΔΔ dark':>10} {'ΔΔ bright':>10} "
              f"{'ρ(C-Δ,L)':>9} {'ρ(Δmode,L)':>11}  Kết luận ΔΔ")
        summary = {}
        for idx, rec in records.items():
            st = stats[idx]
            h, w = rec['delta_mode'].shape
            dd, cd, dm = st['delta_delta'], st['c_delta'], st['delta_mode']
            f = lambda s, k: f"{s[k]:+.4f}" if s is not None else 'N/A'
            print(f"{rec['name']:<16} {f'{w}x{h}':<11} {f(dd, 'spearman'):>9} {f(dd, 'mean_dark'):>10} "
                  f"{f(dd, 'mean_bright'):>10} {f(cd, 'spearman'):>9} {f(dm, 'spearman'):>11}  {verdict(dd)}")

            save_layer_figure(tag, focus_str, idx, rec, st, img_np, out_np,
                              os.path.join(out_dir, f"{rec['name']}_overview.png"))
            if args.save_individual:
                save_individual(rec, os.path.join(out_dir, 'individual'), rec['name'])
            if args.save_npy:
                np.savez_compressed(os.path.join(out_dir, f"{rec['name']}_maps.npz"),
                                    **{k: v for k, v in rec.items() if isinstance(v, np.ndarray)},
                                    luma=st['luma'])
            summary[rec['name']] = {k: v for k, v in st.items() if k != 'luma'}

        with open(os.path.join(out_dir, 'stats.json'), 'w', encoding='utf-8') as fjs:
            json.dump({'checkpoint': ckpt, 'arch': arch, 'focus': focus_str, 'focus_source': src,
                       'image': args.image, 'layers': summary}, fjs, indent=2, ensure_ascii=False)
        print(f"[SAVED] -> {out_dir}")
        del model
        if args.device == 'cuda':
            torch.cuda.empty_cache()

        if args.equiv_test:
            if not arch.startswith('ig'):
                print("[SKIP] Equivalence test chỉ áp dụng cho kiến trúc có IG_Mamba.")
            else:
                equivalence_test(ckpt, arch, df, img, img_np, gt, args.device, args.layers, out_dir, tag)


if __name__ == '__main__':
    main()

import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

# ---------------------------------------------------------------------------
# Tự động phát hiện và import selective_scan_fn từ mamba_ssm.
# Nếu môi trường chưa cài đặt kernel CUDA (ví dụ trên Windows),
# module sẽ tự động chuyển sang pure-PyTorch implementation để không bao giờ bị crash.
# ---------------------------------------------------------------------------
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    HAS_MAMBA_CUDA = True
except ImportError:
    HAS_MAMBA_CUDA = False


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


# ---------------------------------------------------------------------------
# Các module LayerNorm tương thích chuẩn Restormer / MMMamba
# ---------------------------------------------------------------------------
class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type='BiasFree'):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape) == 4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)


# ---------------------------------------------------------------------------
# Gated Depthwise FeedForward Network (GDFN - giống Restormer & MMMamba)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """
    FeedForward Network với Depthwise Convolution và Gating GELU.
    """
    def __init__(self, dim, ffn_expansion_factor=2, bias=False):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


# ---------------------------------------------------------------------------
# Fallback Pure-PyTorch Selective Scan khi không có CUDA kernel
# ---------------------------------------------------------------------------
def selective_scan_ref_pytorch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor = None,
    delta_bias: torch.Tensor = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    b, d, l = u.shape
    n = A.shape[1]

    if delta_bias is not None:
        delta = delta + delta_bias.unsqueeze(0).unsqueeze(-1)
    if delta_softplus:
        delta = F.softplus(delta)

    if B.dim() == 4:
        B = B.squeeze(1)
    if C.dim() == 4:
        C = C.squeeze(1)

    deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)

    ys = []
    x_state = torch.zeros(b, d, n, device=u.device, dtype=u.dtype)
    for i in range(l):
        x_state = deltaA[:, :, i, :] * x_state + deltaB_u[:, :, i, :]
        y_i = torch.einsum('bdn,bn->bd', x_state, C[:, :, i])
        ys.append(y_i)

    y = torch.stack(ys, dim=-1)

    if D is not None:
        y = y + u * D.unsqueeze(0).unsqueeze(-1)

    return y


# ---------------------------------------------------------------------------
# IG_Attention (Khối Mamba Attention với DWConv và Illumination-Guided Scan)
# ---------------------------------------------------------------------------
class IG_Attention(nn.Module):
    """
    Khối Attention cốt lõi:
    - Kế thừa đầy đủ các khối DWConv cho 2 nhánh tương tự mmmamba.py
    - Thay vì ghép xen kẽ token thành chuỗi dài 2HW, nhánh I điều biến Delta và C
      trên chuỗi quét HV độ dài HW (tiết kiệm 50% FLOPs và triệt tiêu loang nhiễu).
    """
    def __init__(
        self,
        d_model: int,
        window_size: int = 2,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        dt_rank: str = "auto",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init: str = "random",
        dt_scale: float = 1.0,
        dt_init_floor: float = 1e-4,
        conv_bias: bool = True,
        bias: bool = False,
        device=None,
        dtype=None,
        enable_padding: bool = True,
        padding_mode: str = 'replicate',
        dark_focus: bool = True,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.enable_padding = enable_padding
        self.padding_mode = padding_mode
        self.dark_focus = dark_focus

        # 1. Tiền xử lý nhánh HV: Linear + DWConv 3x3 + SiLU (giống in_proj_vis & conv2d_vis của mmmamba)
        self.in_proj_hv = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d_hv = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act_hv = nn.SiLU()

        # 2. Tiền xử lý nhánh I: Linear + DWConv 3x3 + SiLU (giống in_proj_inf & conv2d_inf của mmmamba)
        self.in_proj_i = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d_i = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act_i = nn.SiLU()

        # 3. Tham số SSM cho K=4 hướng quét (chuỗi quét dài L = HW)
        self.K = 4

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        self.dt_projs = (
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=self.K, merge=True)

        # 4. Bộ điều biến từ nhánh I (Illumination-Guided Delta & C Modulation)
        # i_delta_mod: Chân ga / Chân phanh ghi vào bộ nhớ
        self.i_delta_mod = nn.Sequential(
            nn.Linear(self.d_inner, self.dt_rank, bias=True, **factory_kwargs),
            nn.Tanh(),
        )
        # i_c_mod: Illumination-Guided Query đọc ra từ bộ nhớ
        self.i_c_mod = nn.Sequential(
            nn.Linear(self.d_inner, self.d_state, bias=True, **factory_kwargs),
            nn.Tanh(),
        )

        # 5. Chuẩn hóa & Chiếu đầu ra nhánh HV
        self.out_norm_hv = nn.LayerNorm(self.d_inner)
        self.out_proj_hv = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @staticmethod
    def _dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def _A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def _D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def pad_to_window_size(self, x: torch.Tensor, w: int):
        B, C, H, W = x.shape
        pad_h = (w - H % w) % w
        pad_w = (w - W % w) % w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=self.padding_mode)
        return x, (H, W), (pad_h, pad_w)

    def _get_four_scans(self, x: torch.Tensor) -> torch.Tensor:
        """
        Quét 2D 4 hướng liên tục:
        1. Trái -> Phải, Trên -> Dưới
        2. Phải -> Trái, Dưới -> Trên
        3. Trên -> Dưới, Trái -> Phải
        4. Dưới -> Trên, Phải -> Trái
        Output: [B, 4, d_inner, L] với L = H * W
        """
        B, C, H, W = x.shape
        L = H * W
        scan1 = x.view(B, C, L)
        scan2 = x.flip(-1).flip(-2).view(B, C, L)
        scan3 = x.transpose(-1, -2).contiguous().view(B, C, L)
        scan4 = x.transpose(-1, -2).flip(-1).flip(-2).contiguous().view(B, C, L)
        return torch.stack([scan1, scan2, scan3, scan4], dim=1)

    def _merge_four_scans(self, y_scans: torch.Tensor, H: int, W: int) -> torch.Tensor:
        """
        Hợp nhất 4 chuỗi quét khôi phục về tensor ảnh 2D [B, C, H, W]
        """
        B, _, C, L = y_scans.shape
        y1 = y_scans[:, 0].view(B, C, H, W)
        y2 = y_scans[:, 1].view(B, C, H, W).flip(-1).flip(-2)
        y3 = y_scans[:, 2].view(B, C, W, H).transpose(-1, -2)
        y4 = y_scans[:, 3].view(B, C, W, H).flip(-1).flip(-2).transpose(-1, -2)
        return y1 + y2 + y3 + y4

    def forward(self, visible: torch.Tensor, infrared: torch.Tensor) -> torch.Tensor:
        """
        visible:  [B, C, H, W] (Nhánh màu sắc HV)
        infrared: [B, C, H, W] (Nhánh độ rọi I)
        """
        orig_H, orig_W = visible.shape[2], visible.shape[3]
        if self.enable_padding:
            visible, (H, W), (pad_h, pad_w) = self.pad_to_window_size(visible, self.window_size)
            infrared, _, _ = self.pad_to_window_size(infrared, self.window_size)
        else:
            H, W = orig_H, orig_W
            pad_h, pad_w = 0, 0

        B, C, H, W = visible.shape
        L = H * W

        # 1. Chiếu kênh qua Linear
        x_vis = rearrange(visible, 'b c h w -> b h w c')
        x_inf = rearrange(infrared, 'b c h w -> b h w c')

        xz_vis = self.in_proj_hv(x_vis)
        x_vis, z_vis = xz_vis.chunk(2, dim=-1)

        xz_inf = self.in_proj_i(x_inf)
        x_inf, _ = xz_inf.chunk(2, dim=-1)

        # 2. Đi qua khối DWConv 3x3 + SiLU (Đặc trưng cốt lõi của mmmamba)
        x_vis = x_vis.permute(0, 3, 1, 2).contiguous()
        x_vis = self.act_hv(self.conv2d_hv(x_vis))

        x_inf = x_inf.permute(0, 3, 1, 2).contiguous()
        x_inf = self.act_i(self.conv2d_i(x_inf))

        # 3. Quét 4 hướng song song: Chuỗi HV và Chuỗi dẫn đường I
        xs_hv = self._get_four_scans(x_vis)  # [B, 4, d_inner, L]
        xs_i = self._get_four_scans(x_inf)   # [B, 4, d_inner, L]

        # 4. Tính toán tham số SSM cơ sở từ nhánh HV
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs_hv,
            self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2
        )

        # 5. Điều biến có hướng dẫn từ nhánh I (Illumination Guidance)
        xs_i_trans = rearrange(xs_i, "b k d l -> (b k l) d")
        i_delta = self.i_delta_mod(xs_i_trans)
        i_delta = rearrange(i_delta, "(b k l) r -> b k r l", b=B, k=self.K, l=L)

        i_c = self.i_c_mod(xs_i_trans)
        i_c = rearrange(i_c, "(b k l) s -> b k s l", b=B, k=self.K, l=L)

        # Cơ chế Dark Focus:
        # Khi dark_focus=True: Vùng tối làm tăng Delta (mở rộng bước lấy mẫu để tập trung học và phục hồi),
        # vùng sáng làm giảm Delta (hãm lại để bảo tồn nguyên vẹn vùng sáng, tránh cháy sáng over-exposure).
        if self.dark_focus:
            i_delta = -i_delta
            i_c = -i_c

        # Áp dụng chân ga/chân phanh cho Delta và Illumination Query cho C
        dts = dts + i_delta
        Cs = Cs + i_c

        # Chiếu Delta lên d_inner
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts,
            self.dt_projs_weight
        )

        # Chuẩn bị định dạng cho Selective Scan
        xs_scan = xs_hv.float().view(B, -1, L)
        dts_scan = dts.float().contiguous().view(B, -1, L)
        Bs_scan = Bs.float().view(B, self.K, -1, L)
        Cs_scan = Cs.float().view(B, self.K, -1, L)

        Ds_vec = self.Ds.float().view(-1)
        As_mat = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias_vec = self.dt_projs_bias.float().view(-1)

        # 6. Thực thi Selective Scan (CUDA Kernel hoặc PyTorch Fallback)
        if HAS_MAMBA_CUDA and xs_scan.is_cuda:
            out_y = selective_scan_fn(
                xs_scan,
                dts_scan,
                As_mat,
                Bs_scan,
                Cs_scan,
                Ds_vec,
                z=None,
                delta_bias=dt_bias_vec,
                delta_softplus=True,
                return_last_state=False,
            ).view(B, self.K, -1, L)
        else:
            out_scans = []
            for k_idx in range(self.K):
                u_k = xs_scan[:, k_idx * self.d_inner : (k_idx + 1) * self.d_inner, :]
                dt_k = dts_scan[:, k_idx * self.d_inner : (k_idx + 1) * self.d_inner, :]
                A_k = As_mat[k_idx * self.d_inner : (k_idx + 1) * self.d_inner, :]
                B_k = Bs_scan[:, k_idx, :, :]
                C_k = Cs_scan[:, k_idx, :, :]
                D_k = Ds_vec[k_idx * self.d_inner : (k_idx + 1) * self.d_inner]
                bias_k = dt_bias_vec[k_idx * self.d_inner : (k_idx + 1) * self.d_inner]

                y_k = selective_scan_ref_pytorch(
                    u_k, dt_k, A_k, B_k, C_k, D_k, delta_bias=bias_k, delta_softplus=True
                )
                out_scans.append(y_k)
            out_y = torch.stack(out_scans, dim=1)

        # 7. Hợp nhất 4 chuỗi quét và kích hoạt gating SiLU
        y_merged = self._merge_four_scans(out_y, H, W)
        y_merged = rearrange(y_merged, "b c h w -> b h w c")

        y_vis = self.out_norm_hv(y_merged)
        y_vis = y_vis * F.silu(z_vis)
        out_vis = self.out_proj_hv(y_vis)
        out_vis = rearrange(out_vis, 'b h w c -> b c h w')

        # 8. Unpad về kích thước ảnh ban đầu
        if self.enable_padding and (pad_h > 0 or pad_w > 0):
            out_vis = out_vis[:, :, :orig_H, :orig_W]

        return out_vis


# ---------------------------------------------------------------------------
# IG_Mamba (Khối hoàn chỉnh với LayerNorm, Attention, FeedForward, Residual)
# ---------------------------------------------------------------------------
class IG_Mamba(nn.Module):
    """
    Module IG_Mamba hoàn chỉnh, có kiến trúc tương đồng 100% với MMMamba:
    - Bao gồm các khối LayerNorm (BiasFree)
    - Khối Attention tích hợp DWConv 3x3 và Illumination-Guided Scanning
    - Khối FeedForward tích hợp DWConv 3x3 và Gated GELU (GDFN)
    - Hai tầng kết nối tắt (Residual Connections)
    """
    def __init__(
        self,
        dim: int = None,
        d_model: int = None,
        window_size: int = 2,
        d_state: int = 16,
        d_conv: int = 3,
        expand: float = 2.0,
        LayerNorm_type: str = 'BiasFree',
        ffn_expansion_factor: float = 2.0,
        enable_padding: bool = True,
        padding_mode: str = 'replicate',
        dark_focus: bool = True,
        **kwargs,
    ):
        super(IG_Mamba, self).__init__()
        if dim is None:
            dim = d_model
        if d_model is None:
            d_model = dim
        self.dim = dim

        # LayerNorm cho nhánh HV và nhánh I
        self.norm_hv = LayerNorm(dim, LayerNorm_type)
        self.norm_i = LayerNorm(dim, LayerNorm_type)
        self.norm_hv_2 = LayerNorm(dim, LayerNorm_type)

        # Khối Attention cốt lõi (với DWConv & Dark Focus)
        self.attn = IG_Attention(
            d_model=dim,
            window_size=window_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            enable_padding=enable_padding,
            padding_mode=padding_mode,
            dark_focus=dark_focus,
            **kwargs,
        )

        # Khối FeedForward (với DWConv và Gated GELU)
        self.ffn = FeedForward(dim, ffn_expansion_factor=ffn_expansion_factor)

    @property
    def dark_focus(self) -> bool:
        return self.attn.dark_focus

    @dark_focus.setter
    def dark_focus(self, val: bool):
        self.attn.dark_focus = val

    def forward(self, x_hv, guide_i=None):
        """
        Hỗ trợ 2 kiểu truyền tham số:
        - Cách 1: model(x_hv, guide_i) -> trả về out_hv
        - Cách 2: model([guide_i, x_hv]) -> trả về [guide_i, out_hv] (tương thích 100% với MMMamba cũ)
        """
        is_list_input = False
        if guide_i is None and isinstance(x_hv, (list, tuple)):
            is_list_input = True
            guide_i, x_hv = x_hv[0], x_hv[1]

        # 1. Khối Attention (DWConv + Mamba Selective Scan) với Residual 1
        norm_hv = self.norm_hv(x_hv)
        norm_i = self.norm_i(guide_i)
        hv_attn = self.attn(norm_hv, norm_i)
        hv = x_hv + hv_attn

        # 2. Khối FeedForward (DWConv + Gated GELU) với Residual 2
        hv_ffn = self.ffn(self.norm_hv_2(hv))
        hv = hv + hv_ffn

        if is_list_input:
            return [guide_i, hv]
        return hv


# Các Alias để tương thích tuyệt đối với các file khác
IlluminationGuidedDeltaMamba = IG_Mamba
MMMamba = IG_Mamba
Attention = IG_Attention

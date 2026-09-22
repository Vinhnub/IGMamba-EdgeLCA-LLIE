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
# 1. Các module LayerNorm tương thích chuẩn Restormer / CIDNet
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
# 2. Gated Depthwise FeedForward Network (GDFN - Restormer style)
# ---------------------------------------------------------------------------
class FeedForward(nn.Module):
    """
    FeedForward Network với Depthwise Convolution 3x3 và Gating GELU.
    """
    def __init__(self, dim, ffn_expansion_factor=2.0, bias=False):
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
# 3. Fallback Pure-PyTorch Selective Scan khi không có CUDA kernel
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
# 4. Standard 2D Mamba Attention (Quét 4 hướng, KHÔNG có Illumination Guidance)
# ---------------------------------------------------------------------------
class StandardMambaAttention(nn.Module):
    """
    Khối Attention 2D Mamba chuẩn (Vanilla / Standard 2D Mamba):
    - Quét 4 hướng song song trên ảnh 2D (L = H * W).
    - KHÔNG sử dụng nhánh dẫn đường ánh sáng (Không có IG / Illumination Guidance).
    - Các tham số SSM (Delta, B, C) được học hoàn toàn nội sinh từ chính đặc trưng đầu vào x.
    - Tích hợp DWConv 3x3 để nắm bắt inductive bias cục bộ trước khi quét SSM toàn cục.
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

        # 1. Chiếu tuyến tính đầu vào: Mở rộng chiều lên d_inner * 2 (x và cổng z)
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        # 2. Depthwise Convolution trích xuất đặc trưng không gian cục bộ
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        # 3. Thiết lập tham số State Space Model (SSM) cho K=4 hướng quét
        self.K = 4

        # Chiếu từ d_inner sang [dt_rank + d_state * 2] (tương ứng Delta, B, C)
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))
        del self.x_proj

        # Chiếu dt từ dt_rank lên d_inner cho 4 hướng
        self.dt_projs = (
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self._dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))
        del self.dt_projs

        # Khởi tạo ma trận A và vector D cho 4 hướng
        self.A_logs = self._A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self._D_init(self.d_inner, copies=self.K, merge=True)

        # 4. Chuẩn hóa & Chiếu đầu ra
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

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
        1. Trái -> Phải, Trên -> Dưới (Horizontal Forward)
        2. Phải -> Trái, Dưới -> Trên (Horizontal Backward)
        3. Trên -> Dưới, Trái -> Phải (Vertical Forward)
        4. Dưới -> Trên, Phải -> Trái (Vertical Backward)
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

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """
        x: [B, C, H, W] (Tensor đặc trưng ảnh đầu vào)
        """
        orig_H, orig_W = x.shape[2], x.shape[3]
        if self.enable_padding:
            x, (H, W), (pad_h, pad_w) = self.pad_to_window_size(x, self.window_size)
        else:
            H, W = orig_H, orig_W
            pad_h, pad_w = 0, 0

        B, C, H, W = x.shape
        L = H * W

        # 1. Chiếu tuyến tính kênh không gian
        x_proj = rearrange(x, 'b c h w -> b h w c')
        xz = self.in_proj(x_proj)
        x_feat, z = xz.chunk(2, dim=-1)

        # 2. Trích xuất đặc trưng cục bộ qua DWConv 3x3 + SiLU
        x_feat = x_feat.permute(0, 3, 1, 2).contiguous()
        x_feat = self.act(self.conv2d(x_feat))

        # 3. Quét 4 hướng song song: [B, 4, d_inner, L]
        xs = self._get_four_scans(x_feat)

        # 4. Dự đoán tham số SSM thuần túy từ x (KHÔNG CÓ IG)
        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l",
            xs,
            self.x_proj_weight
        )
        dts, Bs, Cs = torch.split(
            x_dbl,
            [self.dt_rank, self.d_state, self.d_state],
            dim=2
        )

        # Chiếu Delta lên số chiều d_inner
        dts = torch.einsum(
            "b k r l, k d r -> b k d l",
            dts,
            self.dt_projs_weight
        )

        # Chuẩn bị tensor cho Selective Scan
        xs_scan = xs.float().view(B, -1, L)
        dts_scan = dts.float().contiguous().view(B, -1, L)
        Bs_scan = Bs.float().view(B, self.K, -1, L)
        Cs_scan = Cs.float().view(B, self.K, -1, L)

        Ds_vec = self.Ds.float().view(-1)
        As_mat = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias_vec = self.dt_projs_bias.float().view(-1)

        # 5. Thực thi Selective Scan (CUDA Kernel hoặc PyTorch Fallback)
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

        # 6. Hợp nhất 4 chuỗi quét và kích hoạt cổng SiLU
        y_merged = self._merge_four_scans(out_y, H, W)
        y_merged = rearrange(y_merged, "b c h w -> b h w c")

        y_out = self.out_norm(y_merged)
        y_out = y_out * F.silu(z)
        out = self.out_proj(y_out)
        out = rearrange(out, 'b h w c -> b c h w')

        # 7. Unpad về kích thước ảnh gốc nếu đã pad
        if self.enable_padding and (pad_h > 0 or pad_w > 0):
            out = out[:, :, :orig_H, :orig_W]

        return out


# ---------------------------------------------------------------------------
# 5. Khối Standard Mamba hoàn chỉnh (LayerNorm + Attention + GDFN + Residual)
# ---------------------------------------------------------------------------
class StandardMamba(nn.Module):
    """
    Module Standard Mamba (Không có IG) hoàn chỉnh cho Vision 2D:
    - Bao gồm các khối LayerNorm (BiasFree)
    - Khối 2D Mamba Attention quét 4 hướng song song
    - Khối FeedForward tích hợp DWConv 3x3 và Gated GELU (GDFN)
    - Hai tầng kết nối tắt (Dual Residual Connections)
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
        **kwargs,
    ):
        super(StandardMamba, self).__init__()
        if dim is None:
            dim = d_model
        if d_model is None:
            d_model = dim
        self.dim = dim

        # LayerNorm
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        # Khối Mamba Attention 2D quét 4 hướng (Không có IG)
        self.attn = StandardMambaAttention(
            d_model=dim,
            window_size=window_size,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            enable_padding=enable_padding,
            padding_mode=padding_mode,
            **kwargs,
        )

        # Khối FeedForward (DWConv 3x3 + Gated GELU)
        self.ffn = FeedForward(dim, ffn_expansion_factor=ffn_expansion_factor)

    def forward(self, x, guide=None):
        """
        x: [B, C, H, W]
        guide: Tùy chọn, nếu truyền vào sẽ tự động bỏ qua để tương thích API của CIDNet
        """
        # Nếu truyền danh sách [x, guide] giống giao diện cũ
        if guide is None and isinstance(x, (list, tuple)):
            x = x[0]

        # 1. Attention với Residual 1
        x = x + self.attn(self.norm1(x))

        # 2. FeedForward với Residual 2
        x = x + self.ffn(self.norm2(x))

        return x


# Alias thuận tiện
VanillaMamba = StandardMamba
Mamba2D = StandardMamba
VMambaBlock = StandardMamba


# ==============================================================================
# HÀM KIỂM THỬ (UNIT TEST)
# ==============================================================================
if __name__ == '__main__':
    import sys
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass

    print("=" * 65)
    print(" [TEST] STANDARD 2D MAMBA (KHÔNG CÓ IG) CHO VISION")
    print("=" * 65)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Kernel CUDA mamba_ssm khả dụng: {HAS_MAMBA_CUDA}")

    B, C, H, W = 2, 36, 64, 64
    x = torch.randn(B, C, H, W, device=device, requires_grad=True)

    # Khởi tạo mô hình
    model = StandardMamba(dim=C, d_state=16, expand=2.0).to(device)

    # 1. Test Forward với kích thước chẵn
    out = model(x)
    print(f"\n[1] Test Forward (Kích thước chẵn {H}x{W}):")
    print(f"    - Input shape:  {x.shape}")
    print(f"    - Output shape: {out.shape}")
    assert out.shape == x.shape, "Lỗi kích thước đầu ra không khớp đầu vào!"

    # 2. Test Forward với kích thước lẻ (Kiểm tra cơ chế padding)
    x_odd = torch.randn(B, C, 63, 63, device=device)
    out_odd = model(x_odd)
    print(f"\n[2] Test Forward (Kích thước lẻ 63x63):")
    print(f"    - Input shape:  {x_odd.shape}")
    print(f"    - Output shape: {out_odd.shape}")
    assert out_odd.shape == x_odd.shape, "Lỗi kích thước khi pad kích thước lẻ!"

    # 3. Test Backward Gradient
    loss = out.sum()
    loss.backward()
    print(f"\n[3] Test Backward Gradient:")
    print(f"    - Gradient x_grad norm: {x.grad.norm().item():.4f}")
    assert x.grad is not None, "Lỗi: Không tính được gradient cho đầu vào!"

    # 4. Đếm số lượng tham số
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n[4] Tổng số tham số của StandardMamba (dim={C}): {total_params:,} params")

    print("\n=> ALL UNIT TESTS PASSED SUCCESSFULLY!")

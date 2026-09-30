import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import LayerNorm
from net.LCA import IEL


class SAB(nn.Module):
    """
    Self-Attention Block (MDTA - Multi-Dconv Head Transposed Self-Attention).
    Được thiết kế đồng nhất về số kênh và kiến trúc với CAB (Cross-Attention Block),
    nhưng cả Q, K, V đều được sinh nội suy từ cùng 1 đặc trưng đầu vào x (nhánh I).
    """
    def __init__(self, dim, num_heads, bias=False):
        super(SAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Phép chiếu Q, K, V từ cùng 1 đầu vào x
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(
            dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias
        )
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, *args, **kwargs):
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)

        # Transposed attention trên chiều kênh (Channel Attention)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = F.softmax(attn, dim=-1)

        out = attn @ v
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out


class I_SA(nn.Module):
    """
    Intensity Self-Attention Layer (Dùng để thay thế I_LCA trong Ablation Study).
    - Cấu trúc đối xứng với I_LCA (LayerNorm + Attention + Gated IEL FFN + Residual).
    - KHÔNG nhận nhánh HV làm tương tác chéo (Key/Value), mà chỉ tự chú ý trên chính nhánh I.
    """
    def __init__(self, dim, num_heads, bias=False):
        super(I_SA, self).__init__()
        self.norm = LayerNorm(dim)
        self.gdfn = IEL(dim)
        self.attn = SAB(dim, num_heads, bias=bias)

    def forward(self, x, y=None):
        """
        Tham số y (đặc trưng nhánh HV) được chấp nhận nhưng cố tình bị bỏ qua
        để tương thích 100% chữ ký hàm của I_LCA: module(i_enc, hv_enc).
        """
        x = x + self.attn(self.norm(x))
        x = x + self.gdfn(self.norm(x))
        return x


# Alias: LSA (Lightweight Self-Attention) tương đương 1:1 với LCA (Lightweight Cross-Attention)
LSA = I_SA

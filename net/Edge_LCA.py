import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from net.transformer_utils import LayerNorm
from net.LCA import IEL


class EdgeGuided_CAB(nn.Module):
    """
    Cross Attention Block có hướng dẫn từ bản đồ cạnh (Edge-Guided CAB).
    - x (Query): Đặc trưng nhánh Độ sáng Intensity (I) [B, C, H, W]
    - y (Key, Value): Đặc trưng nhánh Màu sắc Chrominance (HV) [B, C, H, W]
    - edge_map: Bản đồ cạnh cấu trúc [B, 1, H_orig, W_orig] được trích xuất từ kênh I
    
    Cơ chế:
    1. Điều biến không gian Q và K bằng trọng số cạnh (Edge-Weighted Spatial Modulation).
    2. Tính Transposed Cross Attention theo chiều kênh (Channel Covariance).
    3. Cổng lọc không gian (Spatial Edge Gating) ở ngõ ra để triệt tiêu nhiễu vùng phẳng tối.
    """
    def __init__(self, dim, num_heads, bias=False):
        super(EdgeGuided_CAB, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # Phép chiếu Query (từ nhánh I)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)

        # Phép chiếu Key & Value (từ nhánh HV)
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)

        # Hệ số học điều biến cạnh cho Q và K
        self.edge_scale = nn.Parameter(torch.tensor(0.5))

        # Cổng không gian ngõ ra (Edge Gating)
        self.edge_gate = nn.Sequential(
            nn.Conv2d(1, dim, kernel_size=3, padding=1, bias=bias),
            nn.Sigmoid()
        )

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y, edge_map=None):
        b, c, h, w = x.shape

        q = self.q_dwconv(self.q(x))
        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)

        # 1. Điều biến Q và K dựa trên thông tin cạnh
        if edge_map is not None:
            # Tự động co/dãn bản đồ cạnh về đúng kích thước của tầng hiện tại
            if edge_map.shape[-2:] != (h, w):
                edge_scaled = F.interpolate(edge_map, size=(h, w), mode='bilinear', align_corners=False)
            else:
                edge_scaled = edge_map

            # Khuếch đại đặc trưng tại các vị trí có biên cạnh sắc nét
            weight = 1.0 + self.edge_scale * edge_scaled
            q = q * weight
            k = k * weight

        # 2. Transposed Cross Attention (tương quan chéo giữa các kênh)
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

        # 3. Cổng lọc cạnh ở ngõ ra (dập tắt nhiễu ở các vùng tối phẳng không có cạnh)
        if edge_map is not None:
            gate = self.edge_gate(edge_scaled)
            out = out * gate

        return out


class Edge_Guided_I_LCA(nn.Module):
    """
    Khối Lightweight Cross Attention có hướng dẫn cạnh cho nhánh Intensity (I).
    Thay thế cho I_LCA nguyên bản.
    """
    def __init__(self, dim, num_heads, bias=False):
        super(Edge_Guided_I_LCA, self).__init__()
        self.norm = LayerNorm(dim)
        self.ffn = EdgeGuided_CAB(dim, num_heads, bias=bias)
        self.gdfn = IEL(dim)

    def forward(self, x, y, edge_map=None):
        # x: Intensity (I), y: Chrominance (HV), edge_map: Bản đồ cạnh
        x = x + self.ffn(self.norm(x), self.norm(y), edge_map=edge_map)
        x = x + self.gdfn(self.norm(x))
        return x


# Alias để tương thích
I_LCA_Edge = Edge_Guided_I_LCA

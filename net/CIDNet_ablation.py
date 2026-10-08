import torch
import torch.nn as nn
import torch.nn.functional as F
from huggingface_hub import PyTorchModelHubMixin

from net.HVI_transform import RGB_HVI
from net.transformer_utils import NormDownsample, NormUpsample
from net.LCA import HV_LCA, I_LCA
from net.standard_mamba import StandardMamba


class CIDNetAblation(nn.Module, PyTorchModelHubMixin):
    """
    CIDNet Ablation Study Architecture:
    - KHÔNG sử dụng IG_Mamba (chỉ dùng Standard Mamba 2D thuần túy).
    - KHÔNG sử dụng EdgeLCA hay bộ trích xuất cạnh EdgeExtractor (chỉ dùng LCA bình thường).
    
    Hỗ trợ 4 cấu hình cốt lõi theo nhánh HV và nhánh I:
      1. hv_mode='mamba', i_mode='mamba': Cả HV và I đều là Standard Mamba
      2. hv_mode='lca',   i_mode='lca':   Cả HV và I đều là LCA bình thường (CIDNet baseline)
      3. hv_mode='mamba', i_mode='lca':   HV là Standard Mamba, I là LCA bình thường
      4. hv_mode='lca',   i_mode='mamba': I là Standard Mamba, HV là LCA bình thường
    """

    def __init__(
        self,
        hv_mode: str = 'mamba',      # 'mamba' hoặc 'lca'
        i_mode: str = 'mamba',       # 'mamba' hoặc 'lca'
        channels: list = [36, 36, 72, 144],
        heads: list = [1, 2, 4, 8],
        norm: bool = False,
        order: str = 'separable',     # 'separable' (I cập nhật trước rồi hướng dẫn HV) hoặc 'parallel'
        d_state: int = 16,
        expand: float = 2.0,
        enable_padding: bool = True,  # Tự động pad kích thước chia hết cho 8
        **kwargs
    ):
        super(CIDNetAblation, self).__init__()

        hv_mode = hv_mode.lower()
        i_mode = i_mode.lower()
        assert hv_mode in ('mamba', 'lca'), f"hv_mode không hợp lệ: {hv_mode}. Chọn 'mamba' hoặc 'lca'."
        assert i_mode in ('mamba', 'lca'), f"i_mode không hợp lệ: {i_mode}. Chọn 'mamba' hoặc 'lca'."
        assert order in ('separable', 'parallel'), f"order không hợp lệ: {order}. Chọn 'separable' hoặc 'parallel'."

        self.hv_mode = hv_mode
        self.i_mode = i_mode
        self.order = order
        self.enable_padding = enable_padding

        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # ---------------------------------------------------------------------
        # 1. Nhánh Màu sắc (HV_ways) - Trích xuất đặc trưng & Tái tạo
        # ---------------------------------------------------------------------
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # ---------------------------------------------------------------------
        # 2. Nhánh Độ rọi (I_ways) - Trích xuất đặc trưng & Tái tạo
        # ---------------------------------------------------------------------
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        # ---------------------------------------------------------------------
        # 3. Khởi tạo các khối tương tác theo từng Stage (1 đến 6)
        # ---------------------------------------------------------------------
        stage_configs = [
            (ch2, head2),  # Stage 1: Scale 1/2 (Encoder)
            (ch3, head3),  # Stage 2: Scale 1/4 (Encoder)
            (ch4, head4),  # Stage 3: Scale 1/8 (Bottleneck Encoder)
            (ch4, head4),  # Stage 4: Scale 1/8 (Bottleneck Decoder)
            (ch3, head3),  # Stage 5: Scale 1/4 (Decoder)
            (ch2, head2),  # Stage 6: Scale 1/2 (Decoder)
        ]

        self.I_blocks = nn.ModuleList()
        self.HV_blocks = nn.ModuleList()

        for idx, (dim, head) in enumerate(stage_configs, start=1):
            # Khối nhánh I: Standard Mamba hoặc I_LCA thuần túy
            if self.i_mode == 'mamba':
                i_mod = StandardMamba(dim=dim, d_state=d_state, expand=expand)
            else:
                i_mod = I_LCA(dim=dim, num_heads=head)
            self.I_blocks.append(i_mod)
            setattr(self, f'I_block{idx}', i_mod)

            # Khối nhánh HV: Standard Mamba hoặc HV_LCA thuần túy
            if self.hv_mode == 'mamba':
                hv_mod = StandardMamba(dim=dim, d_state=d_state, expand=expand)
            else:
                hv_mod = HV_LCA(dim=dim, num_heads=head)
            self.HV_blocks.append(hv_mod)
            setattr(self, f'HV_block{idx}', hv_mod)

        # Biến đổi không gian màu RGB <-> HVI
        self.trans = RGB_HVI()

    @property
    def dark_focus(self):
        """Giữ tương thích API với train.py mà không gây lỗi."""
        return None

    @dark_focus.setter
    def dark_focus(self, val):
        pass

    def _forward_block(self, i_mod, hv_mod, i_in, hv_in):
        """
        Thực hiện tính toán 1 tầng tương tác giữa nhánh I và HV:
        - separable: I cập nhật trước, sau đó cung cấp làm guide cho HV (nếu HV dùng LCA).
        - parallel: cả I và HV cùng nhận đầu vào ban đầu của tầng đó.
        """
        if self.order == 'separable':
            # 1. Tính toán nhánh I
            if self.i_mode == 'lca':
                i_out = i_mod(i_in, hv_in)
            else:
                i_out = i_mod(i_in)

            # 2. Tính toán nhánh HV với guide là i_out (nếu dùng LCA)
            if self.hv_mode == 'lca':
                hv_out = hv_mod(hv_in, i_out)
            else:
                hv_out = hv_mod(hv_in)
        else:
            # 1. Tính toán song song (Parallel)
            if self.i_mode == 'lca':
                i_out = i_mod(i_in, hv_in)
            else:
                i_out = i_mod(i_in)

            if self.hv_mode == 'lca':
                hv_out = hv_mod(hv_in, i_in)
            else:
                hv_out = hv_mod(hv_in)

        return i_out, hv_out

    def forward(self, x, return_feats=False):
        orig_H, orig_W = x.shape[2], x.shape[3]
        pad_h, pad_w = 0, 0
        if self.enable_padding:
            factor = 8
            pad_h = (factor - orig_H % factor) % factor
            pad_w = (factor - orig_W % factor) % factor
            if pad_h > 0 or pad_w > 0:
                x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')

        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)

        # Trích xuất đặc trưng nông
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)

        i_jump0 = i_enc0
        hv_jump0 = hv_0

        # =====================================================================
        # ENCODER
        # =====================================================================
        # Stage 1 (Scale 1/2)
        i_enc2, hv_2 = self._forward_block(self.I_block1, self.HV_block1, i_enc1, hv_1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        # Stage 2 (Scale 1/4)
        i_enc3, hv_3 = self._forward_block(self.I_block2, self.HV_block2, i_enc2, hv_2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        # Stage 3 - Bottleneck Encoder (Scale 1/8)
        i_enc4, hv_4 = self._forward_block(self.I_block3, self.HV_block3, i_enc3, hv_3)

        # Stage 4 - Bottleneck Decoder (Scale 1/8)
        i_dec4, hv_4 = self._forward_block(self.I_block4, self.HV_block4, i_enc4, hv_4)

        # =====================================================================
        # DECODER
        # =====================================================================
        # Stage 5 (Scale 1/4)
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)

        i_dec2, hv_2 = self._forward_block(self.I_block5, self.HV_block5, i_dec3, hv_3)

        # Stage 6 (Scale 1/2)
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1, hv_1 = self._forward_block(self.I_block6, self.HV_block6, i_dec2, hv_2)

        # =====================================================================
        # TÁI TẠO ĐẦU RA (RECONSTRUCTION)
        # =====================================================================
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        # Cắt bỏ phần padding nếu đã pad
        if self.enable_padding and (pad_h > 0 or pad_w > 0):
            output_rgb = output_rgb[:, :, :orig_H, :orig_W]

        if return_feats:
            feats = {
                'i_enc2': i_enc2,
                'hv_2': hv_2,
                'i_enc3': i_enc3,
                'hv_3': hv_3,
                'i_enc4': i_enc4,
                'hv_4': hv_4,
                'i_dec2': i_dec2,
                'i_dec1': i_dec1,
            }
            return output_rgb, feats

        return output_rgb

    def HVIT(self, x):
        return self.trans.HVIT(x)

    def RGB2YCrCb(self, x):
        return self.trans.RGB2YCrCb(x)


# =============================================================================
# 4 LỚP TƯỜNG MINH CHO 4 TRƯỜNG HỢP ABLATION STUDY
# =============================================================================

class CIDNet_HV_Mamba_I_Mamba(CIDNetAblation):
    """Trường hợp 1: Cả HV và I đều là Standard Mamba."""
    def __init__(self, **kwargs):
        super(CIDNet_HV_Mamba_I_Mamba, self).__init__(hv_mode='mamba', i_mode='mamba', **kwargs)


class CIDNet_HV_LCA_I_LCA(CIDNetAblation):
    """Trường hợp 2: Cả HV và I đều là LCA bình thường (Baseline CIDNet)."""
    def __init__(self, **kwargs):
        super(CIDNet_HV_LCA_I_LCA, self).__init__(hv_mode='lca', i_mode='lca', **kwargs)


class CIDNet_HV_Mamba_I_LCA(CIDNetAblation):
    """Trường hợp 3: HV là Standard Mamba, I là LCA bình thường."""
    def __init__(self, **kwargs):
        super(CIDNet_HV_Mamba_I_LCA, self).__init__(hv_mode='mamba', i_mode='lca', **kwargs)


class CIDNet_HV_LCA_I_Mamba(CIDNetAblation):
    """Trường hợp 4: I là Standard Mamba, HV là LCA bình thường."""
    def __init__(self, **kwargs):
        super(CIDNet_HV_LCA_I_Mamba, self).__init__(hv_mode='lca', i_mode='mamba', **kwargs)


# Từ điển metadata và builder cho các trường hợp
ABLATION_CONFIGS = {
    'mamba_mamba': {
        'case_id': 1,
        'tag': 'mamba_mamba',
        'name': 'Case 1: Both HV and I are Standard Mamba',
        'desc': 'HV: Standard Mamba, I: Standard Mamba (No EdgeLCA, No IGMamba)',
        'hv_mode': 'mamba',
        'i_mode': 'mamba',
        'cls': CIDNet_HV_Mamba_I_Mamba,
    },
    'lca_lca': {
        'case_id': 2,
        'tag': 'lca_lca',
        'name': 'Case 2: Both HV and I are Standard LCA',
        'desc': 'HV: Standard LCA, I: Standard LCA (Standard CIDNet Baseline)',
        'hv_mode': 'lca',
        'i_mode': 'lca',
        'cls': CIDNet_HV_LCA_I_LCA,
    },
    'mamba_lca': {
        'case_id': 3,
        'tag': 'mamba_lca',
        'name': 'Case 3: HV is Standard Mamba, I is Standard LCA',
        'desc': 'HV: Standard Mamba, I: Standard LCA',
        'hv_mode': 'mamba',
        'i_mode': 'lca',
        'cls': CIDNet_HV_Mamba_I_LCA,
    },
    'lca_mamba': {
        'case_id': 4,
        'tag': 'lca_mamba',
        'name': 'Case 4: I is Standard Mamba, HV is Standard LCA',
        'desc': 'HV: Standard LCA, I: Standard Mamba',
        'hv_mode': 'lca',
        'i_mode': 'mamba',
        'cls': CIDNet_HV_LCA_I_Mamba,
    },
}


def build_ablation_model(case_identifier, **kwargs) -> CIDNetAblation:
    """
    Factory function khởi tạo mô hình theo case:
    - case_identifier có thể là:
        + Số nguyên: 1, 2, 3, 4
        + Chuỗi: '1', '2', '3', '4'
        + Tag: 'mamba_mamba', 'lca_lca', 'mamba_lca', 'lca_mamba'
        + Tên viết tắt: 'both_mamba', 'both_lca', 'hv_mamba_i_lca', 'i_mamba_hv_lca'
    """
    alias_map = {
        '1': 'mamba_mamba',
        1: 'mamba_mamba',
        'both_mamba': 'mamba_mamba',
        'case1': 'mamba_mamba',
        'mamba_mamba': 'mamba_mamba',

        '2': 'lca_lca',
        2: 'lca_lca',
        'both_lca': 'lca_lca',
        'case2': 'lca_lca',
        'lca_lca': 'lca_lca',

        '3': 'mamba_lca',
        3: 'mamba_lca',
        'hv_mamba_i_lca': 'mamba_lca',
        'case3': 'mamba_lca',
        'mamba_lca': 'mamba_lca',

        '4': 'lca_mamba',
        4: 'lca_mamba',
        'i_mamba_hv_lca': 'lca_mamba',
        'case4': 'lca_mamba',
        'lca_mamba': 'lca_mamba',
    }

    key = alias_map.get(case_identifier)
    if key is None:
        raise ValueError(
            f"Case identifier '{case_identifier}' không hợp lệ. "
            f"Chọn một trong: 1 (mamba_mamba), 2 (lca_lca), 3 (mamba_lca), 4 (lca_mamba)"
        )

    model_cls = ABLATION_CONFIGS[key]['cls']
    return model_cls(**kwargs)

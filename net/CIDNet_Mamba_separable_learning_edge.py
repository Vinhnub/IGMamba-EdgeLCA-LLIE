import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from net.HVI_transform import RGB_HVI
from net.transformer_utils import NormDownsample, NormUpsample
from net.Edge_LCA import Edge_Guided_I_LCA
from net.IG_Mamba import IG_Mamba
from net.edge_filter import EdgeExtractor


class CIDNet(nn.Module, PyTorchModelHubMixin):
    """
    CIDNet phiên bản Separable Learning kết hợp IG_Mamba và Dò cạnh (Edge Detector):
    - Nhánh Intensity (I) được trích xuất bản đồ cạnh thông qua EdgeExtractor (Sobel + Laplacian + Conv1x1).
    - Bản đồ cạnh điều phối Cross Attention trong Edge_Guided_I_LCA (tăng cường viền và dập tắt nhiễu vùng tối).
    - Nhánh I sau khi được tinh chỉnh và làm rõ biên sẽ dẫn đường cho IG_Mamba phục hồi màu sắc nhánh HV.
    """
    def __init__(self, 
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 dark_focus=True):
        super(CIDNet, self).__init__()
        
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads
        
        # 1. Bộ trích xuất cạnh cấu trúc (từ kênh Intensity - 1 kênh)
        self.edge_extractor = EdgeExtractor(in_channels=1)

        # 2. Nhánh Màu sắc (HV_ways)
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
        
        # 3. Nhánh Độ rọi (I_ways)
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
        
        # 4. Các khối Edge-Guided I_LCA (Cross Attention có điều biến cạnh)
        self.I_LCA1 = Edge_Guided_I_LCA(ch2, head2)
        self.I_LCA2 = Edge_Guided_I_LCA(ch3, head3)
        self.I_LCA3 = Edge_Guided_I_LCA(ch4, head4)
        self.I_LCA4 = Edge_Guided_I_LCA(ch4, head4)
        self.I_LCA5 = Edge_Guided_I_LCA(ch3, head3)
        self.I_LCA6 = Edge_Guided_I_LCA(ch2, head2)

        # 5. Các khối IG_Mamba (Illumination-Guided Delta Modulation Mamba)
        self.IG_Mamba_1 = IG_Mamba(ch2, dark_focus=dark_focus)
        self.IG_Mamba_2 = IG_Mamba(ch3, dark_focus=dark_focus)
        self.IG_Mamba_3 = IG_Mamba(ch4, dark_focus=dark_focus)
        self.IG_Mamba_4 = IG_Mamba(ch4, dark_focus=dark_focus)
        self.IG_Mamba_5 = IG_Mamba(ch3, dark_focus=dark_focus)
        self.IG_Mamba_6 = IG_Mamba(ch2, dark_focus=dark_focus)
        
        self.trans = RGB_HVI()

    @property
    def dark_focus(self) -> bool:
        return self.IG_Mamba_1.dark_focus

    @dark_focus.setter
    def dark_focus(self, val: bool):
        for m in self.modules():
            if isinstance(m, IG_Mamba):
                m.dark_focus = val
        
    def forward(self, x, return_feats=False):
        dtypes = x.dtype
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2:3, :, :].to(dtypes)  # Kênh Intensity [B, 1, H, W]

        # =====================================================================
        # BƯỚC 1: TRÍCH XUẤT BẢN ĐỒ CẠNH TỪ KÊNH I (1 lần ở đầu mạng)
        # =====================================================================
        edge_map = self.edge_extractor(i).to(dtypes)  # [B, 1, H, W]

        # Trích xuất đặc trưng nông
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)

        i_jump0 = i_enc0
        hv_jump0 = hv_0

        # =====================================================================
        # BƯỚC 2: ENCODER (Học phân tách có hướng dẫn cạnh và Mamba)
        # =====================================================================
        # Tầng Encoder 1 (Scale 1/2)
        i_enc2 = self.I_LCA1(i_enc1, hv_1, edge_map=edge_map)
        hv_2 = self.IG_Mamba_1(hv_1, i_enc2)
        v_jump1 = i_enc2
        hv_jump1 = hv_2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)
        
        # Tầng Encoder 2 (Scale 1/4)
        i_enc3 = self.I_LCA2(i_enc2, hv_2, edge_map=edge_map)
        hv_3 = self.IG_Mamba_2(hv_2, i_enc3)
        v_jump2 = i_enc3
        hv_jump2 = hv_3
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)
        
        # Tầng Bottleneck Encoder (Scale 1/8)
        i_enc4 = self.I_LCA3(i_enc3, hv_3, edge_map=edge_map)
        hv_4 = self.IG_Mamba_3(hv_3, i_enc4)
        
        # Tầng Bottleneck Decoder (Scale 1/8)
        i_dec4 = self.I_LCA4(i_enc4, hv_4, edge_map=edge_map)
        hv_4 = self.IG_Mamba_4(hv_4, i_dec4)
        
        # =====================================================================
        # BƯỚC 3: DECODER
        # =====================================================================
        # Tầng Decoder 2 (Scale 1/4)
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)

        i_dec2 = self.I_LCA5(i_dec3, hv_3, edge_map=edge_map)
        hv_2 = self.IG_Mamba_5(hv_3, i_dec2)
        
        # Tầng Decoder 1 (Scale 1/2)
        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)
        
        i_dec1 = self.I_LCA6(i_dec2, hv_2, edge_map=edge_map)
        hv_1 = self.IG_Mamba_6(hv_2, i_dec1)

        # =====================================================================
        # BƯỚC 4: TÁI TẠO ĐẦU RA (RECONSTRUCTION)
        # =====================================================================
        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)
        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)
        
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        output_rgb = self.trans.PHVIT(output_hvi)

        if return_feats:
            feats = {
                'edge_map': edge_map,
                'i_enc2': i_enc2,
                'hv_2': hv_2,
                'i_enc3': i_enc3,
                'hv_3': hv_3,
                'i_enc4': i_enc4,
                'hv_4': hv_4,
                'i_dec2': i_dec2,
                'i_dec1': i_dec1
            }
            return output_rgb, feats

        return output_rgb
    
    def HVIT(self, x):
        return self.trans.HVIT(x)
    
    def RGB2YCrCb(self, x):
        return self.trans.RGB2YCrCb(x)

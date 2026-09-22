import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.vgg_arch import VGGFeatureExtractor, Registry
from loss.loss_utils import *


_reduction_modes = ['none', 'mean', 'sum']

class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * l1_loss(
            pred, target, weight, reduction=self.reduction)
        
class L2Loss(nn.Module):
    """L2 (mean squared error, MSE) loss.

    Args:
        loss_weight (float): Loss weight for L2 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L2Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. '
                             f'Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise
                weights. Default: None.
        """
        return self.loss_weight * mse_loss(
            pred, target, weight, reduction=self.reduction)
        
        
        
class EdgeLoss(nn.Module):
    def __init__(self,loss_weight=1.0, reduction='mean'):
        super(EdgeLoss, self).__init__()
        k = torch.Tensor([[.05, .25, .4, .25, .05]])
        self.kernel = torch.matmul(k.t(),k).unsqueeze(0).repeat(3,1,1,1).cuda()

        self.weight = loss_weight
        
    def conv_gauss(self, img):
        n_channels, _, kw, kh = self.kernel.shape
        img = F.pad(img, (kw//2, kh//2, kw//2, kh//2), mode='replicate')
        return F.conv2d(img, self.kernel, groups=n_channels)

    def laplacian_kernel(self, current):
        filtered    = self.conv_gauss(current)
        down        = filtered[:,:,::2,::2]
        new_filter  = torch.zeros_like(filtered)
        new_filter[:,:,::2,::2] = down*4
        filtered    = self.conv_gauss(new_filter)
        diff = current - filtered
        return diff

    def forward(self, x, y):
        loss = mse_loss(self.laplacian_kernel(x), self.laplacian_kernel(y))
        return loss*self.weight


class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculting losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=True,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.L2loss()
        elif self.criterion_type == 'mse':
            self.criterion = torch.nn.MSELoss(reduction='mean')
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss




class SSIM(torch.nn.Module):
    def __init__(self, window_size=11, size_average=True,weight=1.):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)
        self.weight = weight

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return (1. - map_ssim(img1, img2, window, self.window_size, channel, self.size_average)) * self.weight


_reduction_modes = ['none', 'mean', 'sum']

class DarkFocusedGradientLoss(nn.Module):
    """
    Dark-Focused / Bright-Focused Spatial Gradient Difference Loss (Tên cũ: RegionLSGDLoss / LSGD).

    Chức năng cốt lõi:
    - So sánh sai khác gradient bậc một (Spatial Gradient Difference) giữa ảnh dự đoán (pred)
      và ảnh mục tiêu (gt) theo cả hai hướng ngang (x) và dọc (y) để bảo toàn chi tiết và biên cạnh.
    
    Cơ chế Tập trung theo vùng sáng/tối (Focus Mechanism):
    - Khi dark_focus=True: Tập trung PHẠT NẶNG NẾU SAI LỆCH Ở VÙNG TỐI.
      Trọng số: weight = (1.0 - luminance)^gamma
      (Vùng tối có trọng số phạt cao, vùng sáng có trọng số phạt thấp).
    - Khi dark_focus=False: Tập trung PHẠT NẶNG NẾU SAI LỆCH Ở VÙNG SÁNG.
      Trọng số: weight = (luminance)^gamma
      (Vùng sáng có trọng số phạt cao, vùng tối có trọng số phạt thấp).
    - Khi dark_focus=None: Phạt đồng đều trên toàn bộ ảnh (không dùng trọng số phân vùng).
    """

    def __init__(
        self,
        loss_weight=1.0,
        reduction='mean',
        dark_focus=True,
        dark_power=1.0,
        eps=1e-6,
        use_dark_weight=None,  # Hỗ trợ tương thích ngược với code cũ
    ):
        super(DarkFocusedGradientLoss, self).__init__()

        if reduction not in _reduction_modes:
            raise ValueError(
                f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}'
            )

        self.loss_weight = loss_weight
        self.reduction = reduction
        # Nếu có truyền use_dark_weight từ code cũ, ưu tiên gán cho dark_focus
        self.dark_focus = use_dark_weight if use_dark_weight is not None else dark_focus
        self.dark_power = dark_power
        self.eps = eps

    def compute_spatial_gradient(self, img):
        """
        Tính gradient không gian bậc nhất theo hướng x và y.
        img: [B, C, H, W]
        Trả về:
            grad_x: [B, C, H, W - 1]
            grad_y: [B, C, H - 1, W]
        """
        grad_x = img[:, :, :, 1:] - img[:, :, :, :-1]
        grad_y = img[:, :, 1:, :] - img[:, :, :-1, :]
        return grad_x, grad_y

    def compute_weight_map(self, gt, is_hvi=False):
        """
        Tính toán bản đồ trọng số:
        - Nếu dark_focus=True:  Phạt vùng tối  -> weight = (1.0 - luminance) ^ dark_power
        - Nếu dark_focus=False: Phạt vùng sáng -> weight = (luminance) ^ dark_power
        """
        channels = gt.shape[1]

        if channels == 3:
            if is_hvi:
                # Trong không gian HVI, kênh thứ 3 là kênh độ rọi I
                luminance = gt[:, 2:3]
            else:
                # Trong không gian RGB, tính độ chói theo công thức chuẩn BT.601
                luminance = (
                    0.299 * gt[:, 0:1]
                    + 0.587 * gt[:, 1:2]
                    + 0.114 * gt[:, 2:3]
                )
        elif channels == 1:
            luminance = gt
        else:
            raise ValueError(f'Unsupported channel size for weight map: {channels}')

        # Đảm bảo độ chói nằm trong khoảng [0, 1]
        luminance = torch.clamp(luminance, min=0.0, max=1.0)

        # ----------------------------------------------------------------------
        # PHÂN ĐỊNH TRỌNG SỐ THEO DARK_FOCUS:
        # - dark_focus = True:  Phạt nặng nếu sai ở VÙNG TỐI (1.0 - luminance)
        # - dark_focus = False: Phạt nặng nếu sai ở VÙNG SÁNG (luminance)
        # ----------------------------------------------------------------------
        if self.dark_focus is True:
            target_weight = 1.0 - luminance
        elif self.dark_focus is False:
            target_weight = luminance
        else:
            # Nếu dark_focus là None: trọng số bằng 1.0 đồng đều
            return torch.ones_like(luminance)

        if self.dark_power != 1.0:
            target_weight = target_weight.pow(self.dark_power)

        weight = target_weight.clamp(min=self.eps, max=1.0)
        return weight

    def reduction_fn(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        return x

    def forward(self, pred, gt, is_hvi=False):
        """
        pred: [B, C, H, W] - Ảnh đầu ra dự đoán của mô hình (RGB hoặc HVI)
        gt:   [B, C, H, W] - Ảnh mục tiêu Ground-Truth (RGB hoặc HVI)
        is_hvi: bool       - Đặt True nếu ảnh đầu vào đang ở không gian màu HVI
        """
        # 1. Tính toán gradient không gian cho pred và gt
        pred_gx, pred_gy = self.compute_spatial_gradient(pred)
        gt_gx, gt_gy = self.compute_spatial_gradient(gt)

        # 2. Tính sai lệch gradient bậc một (L1 Gradient Difference)
        diff_x = torch.abs(pred_gx - gt_gx)
        diff_y = torch.abs(pred_gy - gt_gy)

        # 3. Áp dụng cơ chế trọng số (Dark Focus hoặc Bright Focus)
        if self.dark_focus is not None:
            weight = self.compute_weight_map(gt, is_hvi=is_hvi)
            weight_x = weight[:, :, :, 1:]
            weight_y = weight[:, :, 1:, :]

            diff_x = diff_x * weight_x
            diff_y = diff_y * weight_y

        # 4. Gom nhóm theo reduction và nhân loss weight
        loss_x = self.reduction_fn(diff_x)
        loss_y = self.reduction_fn(diff_y)

        loss = loss_x + loss_y
        return loss * self.loss_weight


# Alias giữ nguyên tương thích 100% với toàn bộ codebase cũ (train.py, train_ddp.py, eval.py)
RegionLSGDLoss = DarkFocusedGradientLoss
LSGDLoss = DarkFocusedGradientLoss
DarkFocusedSpatialGradientLoss = DarkFocusedGradientLoss



class ExposureControlLoss(nn.Module):
    """
    Exposure Control Loss (commonly used in Zero-DCE).
    Ép mô hình nâng độ sáng trung bình của các vùng (patch) lên một mức cố định (ví dụ 0.6).
    """
    def __init__(self, patch_size=16, mean_val=0.6, loss_weight=1.0):
        super(ExposureControlLoss, self).__init__()
        # Dùng Average Pooling để tính trung bình độ sáng của từng vùng kích thước patch_size x patch_size
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
        self.loss_weight = loss_weight

    def forward(self, pred):
        """
        Args:
            pred (Tensor): Ảnh dự đoán (RGB) hoặc kênh Độ sáng (I). Shape: [B, C, H, W]
        """
        # Tính cường độ sáng trung bình của từng vùng
        pred_mean = self.pool(pred)
        
        # Phạt nếu độ sáng trung bình của vùng đó khác với mean_val (mức sáng chuẩn)
        loss = torch.mean(torch.abs(pred_mean - self.mean_val))
        
        return loss * self.loss_weight
"""
Custom Loss Functions for Exposure and White Balance enhancement
Designed to guide the model toward better brightness and color correction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ExposureLoss(nn.Module):
    """Loss for encouraging properly exposed images"""
    def __init__(self, target_luminance=0.4, weight=1.0):
        super(ExposureLoss, self).__init__()
        self.target_luminance = target_luminance
        self.weight = weight
        
    def calculate_luminance(self, img):
        """Convert RGB to luminance using standard weights"""
        luminance = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]
        return luminance
    
    def forward(self, img_out, img_gt=None):
        """Calculate exposure loss"""
        luminance_out = self.calculate_luminance(img_out)
        avg_lum = torch.mean(luminance_out)
        
        lum_loss = torch.abs(avg_lum - self.target_luminance)
        hist_loss = self._histogram_balance_loss(luminance_out)
        total_loss = lum_loss + 0.1 * hist_loss
        
        if img_gt is not None:
            luminance_gt = self.calculate_luminance(img_gt)
            gt_match_loss = F.l1_loss(luminance_out, luminance_gt)
            total_loss = total_loss + 0.5 * gt_match_loss
        
        return self.weight * total_loss
    
    def _histogram_balance_loss(self, luminance):
        """Encourage histogram to use full dynamic range"""
        # Flatten to 1D
        lum_flat = luminance.view(-1)
        
        # Calculate histogram entropy (higher entropy = better distribution)
        # Using histogram variance as proxy
        bins = torch.clamp(lum_flat * 255, 0, 254).long()
        hist = torch.bincount(bins, minlength=256).float() + 1  # +1 to avoid log(0)
        hist = hist / hist.sum()
        
        # Entropy: -sum(p * log(p))
        entropy = -(hist * torch.log(hist)).sum()
        
        # We want higher entropy (meaning well-distributed), so loss = -entropy
        return -entropy


class WhiteBalanceLoss(nn.Module):
    """Loss for white balance correction"""
    def __init__(self, weight=1.0):
        super(WhiteBalanceLoss, self).__init__()
        self.weight = weight
        
    def forward(self, img_out, img_gt=None):
        """Calculate white balance loss"""
        r_mean = torch.mean(img_out[:, 0])
        g_mean = torch.mean(img_out[:, 1])
        b_mean = torch.mean(img_out[:, 2])
        
        wb_loss = torch.abs(r_mean - g_mean) + torch.abs(g_mean - b_mean) + torch.abs(b_mean - r_mean)
        
        r_channel = img_out[:, 0:1]
        g_channel = img_out[:, 1:2]
        b_channel = img_out[:, 2:3]
        
        max_channel = torch.max(torch.stack([r_channel, g_channel, b_channel], dim=0), dim=0)[0]
        
        rg_diff = torch.abs(r_channel - g_channel)
        gb_diff = torch.abs(g_channel - b_channel)
        rb_diff = torch.abs(r_channel - b_channel)
        
        local_balance_loss = torch.mean(rg_diff + gb_diff + rb_diff)
        total_loss = wb_loss + 0.1 * local_balance_loss
        
        if img_gt is not None:
            r_mean_gt = torch.mean(img_gt[:, 0])
            g_mean_gt = torch.mean(img_gt[:, 1])
            b_mean_gt = torch.mean(img_gt[:, 2])
            
            gt_match_loss = (torch.abs(r_mean - r_mean_gt) + 
                           torch.abs(g_mean - g_mean_gt) + 
                           torch.abs(b_mean - b_mean_gt))
            total_loss = total_loss + 0.5 * gt_match_loss
        
        return self.weight * total_loss


class ContrastLoss(nn.Module):
    """Loss to improve local contrast"""
    def __init__(self, weight=1.0):
        super(ContrastLoss, self).__init__()
        self.weight = weight
        
    def forward(self, img_out, img_gt=None):
        """Calculate contrast loss"""
        gray_out = 0.299 * img_out[:, 0:1] + 0.587 * img_out[:, 1:2] + 0.114 * img_out[:, 2:3]
        
        kernel = torch.ones(1, 1, 3, 3, device=gray_out.device) / 9.0
        local_mean = F.conv2d(gray_out, kernel, padding=1)
        local_sq_mean = F.conv2d(gray_out ** 2, kernel, padding=1)
        local_var = local_sq_mean - local_mean ** 2
        
        contrast_loss = -torch.mean(torch.sqrt(torch.clamp(local_var, min=1e-8)))
        
        if img_gt is not None:
            gray_gt = 0.299 * img_gt[:, 0:1] + 0.587 * img_gt[:, 1:2] + 0.114 * img_gt[:, 2:3]
            local_mean_gt = F.conv2d(gray_gt, kernel, padding=1)
            local_sq_mean_gt = F.conv2d(gray_gt ** 2, kernel, padding=1)
            local_var_gt = local_sq_mean_gt - local_mean_gt ** 2
            
            var_l1_loss = F.l1_loss(local_var, local_var_gt)
            contrast_loss = contrast_loss + 0.5 * var_l1_loss
        
        return self.weight * contrast_loss


class ColorConstancyLoss(nn.Module):
    """Loss based on color constancy theory"""
    def __init__(self, weight=1.0):
        super(ColorConstancyLoss, self).__init__()
        self.weight = weight
        
    def forward(self, img_out, img_gt=None):
        """Calculate color constancy loss"""
        img_log = torch.log(torch.clamp(img_out, min=1e-8))
        
        r_log = img_log[:, 0]
        g_log = img_log[:, 1]
        b_log = img_log[:, 2]
        
        r_mean = torch.mean(r_log)
        g_mean = torch.mean(g_log)
        b_mean = torch.mean(b_log)
        
        illuminant = torch.stack([r_mean, g_mean, b_mean])
        illuminant_loss = torch.std(illuminant)
        
        if img_gt is not None:
            img_gt_log = torch.log(torch.clamp(img_gt, min=1e-8))
            r_gt = img_gt_log[:, 0]
            g_gt = img_gt_log[:, 1]
            b_gt = img_gt_log[:, 2]
            
            r_mean_gt = torch.mean(r_gt)
            g_mean_gt = torch.mean(g_gt)
            b_mean_gt = torch.mean(b_gt)
            
            illuminant_gt = torch.stack([r_mean_gt, g_mean_gt, b_mean_gt])
            illuminant_loss = F.l1_loss(illuminant, illuminant_gt)
        
        return self.weight * illuminant_loss


class ExposureWhiteBalanceLoss(nn.Module):
    """Combined loss for exposure and white balance"""
    def __init__(self, 
                 exposure_weight=1.0,
                 wb_weight=1.0,
                 contrast_weight=0.2,
                 color_constancy_weight=0.1):
        super(ExposureWhiteBalanceLoss, self).__init__()
        
        self.exposure_loss = ExposureLoss(weight=exposure_weight)
        self.wb_loss = WhiteBalanceLoss(weight=wb_weight)
        self.contrast_loss = ContrastLoss(weight=contrast_weight)
        self.color_constancy_loss = ColorConstancyLoss(weight=color_constancy_weight)
        
    def forward(self, img_out, img_gt=None):
        """Calculate combined exposure and white balance loss"""
        losses = {}
        
        losses['exposure'] = self.exposure_loss(img_out, img_gt)
        losses['white_balance'] = self.wb_loss(img_out, img_gt)
        losses['contrast'] = self.contrast_loss(img_out, img_gt)
        losses['color_constancy'] = self.color_constancy_loss(img_out, img_gt)
        
        total_loss = sum(losses.values())
        losses['total'] = total_loss
        
        return total_loss, losses


class CombinedLoss(nn.Module):
    """Combined loss: photometric + exposure/white balance"""
    def __init__(self,
                 exposure_wb_weight=0.5,
                 photometric_weight=0.5,
                 exposure_weight=1.0,
                 wb_weight=1.0):
        super(CombinedLoss, self).__init__()
        
        self.exposure_wb_loss = ExposureWhiteBalanceLoss(
            exposure_weight=exposure_weight,
            wb_weight=wb_weight
        )
        self.exposure_wb_weight = exposure_wb_weight
        self.photometric_weight = photometric_weight
        
    def forward(self, img_out, img_gt):
        """Calculate combined photometric and exposure/WB loss"""
        losses = {}
        
        exp_wb_loss, exp_wb_losses = self.exposure_wb_loss(img_out, img_gt)
        losses.update({f'exposure_wb_{k}': v for k, v in exp_wb_losses.items()})
        
        photometric_loss = F.l1_loss(img_out, img_gt)
        losses['photometric_l1'] = photometric_loss
        
        total_loss = (self.exposure_wb_weight * exp_wb_loss + 
                     self.photometric_weight * photometric_loss)
        losses['total'] = total_loss
        
        return total_loss, losses


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    img_out = torch.rand(batch_size, 3, 256, 256, device=device)
    img_gt = torch.rand(batch_size, 3, 256, 256, device=device)
    
    print("Testing individual losses...")
    exp_loss = ExposureLoss()
    print(f"Exposure: {exp_loss(img_out, img_gt).item():.4f}")
    
    wb_loss = WhiteBalanceLoss()
    print(f"White Balance: {wb_loss(img_out, img_gt).item():.4f}")
    
    contrast_loss = ContrastLoss()
    print(f"Contrast: {contrast_loss(img_out, img_gt).item():.4f}")
    
    cc_loss = ColorConstancyLoss()
    print(f"Color Constancy: {cc_loss(img_out, img_gt).item():.4f}")
    
    print("\nTesting combined losses...")
    combined_loss = ExposureWhiteBalanceLoss()
    total_loss, loss_dict = combined_loss(img_out, img_gt)
    print(f"Combined: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
    
    print("\nTesting with ground truth...")
    final_loss = CombinedLoss()
    total_loss, loss_dict = final_loss(img_out, img_gt)
    print(f"Final: {total_loss.item():.4f}")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")

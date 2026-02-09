"""
Exposure and White Balance Enhancement Module
Designed to handle both underexposure and overexposure correction
Along with white balance (color temperature) adjustment
"""
import torch
import torch.nn as nn
import numpy as np
from timm.models.layers import trunc_normal_
import math


class ExposureAnalyzer(nn.Module):
    """
    Analyzer module to predict exposure level and required correction direction
    Handles both underexposure (< 0.5 luminance) and overexposure (> 0.7 luminance)
    """
    def __init__(self, dim=64):
        super(ExposureAnalyzer, self).__init__()
        self.dim = dim
        
        # Global pooling to analyze overall luminance
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Exposure estimation network
        self.exposure_predictor = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),  # Output: exposure level (-1 to +1)
        )
        
        # Exposure direction classifier (underexposed vs overexposed)
        self.direction_predictor = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 2),  # Output: [under, over] scores
        )
        
    def forward(self, img):
        """Predict exposure level and correction direction"""
        b, c, h, w = img.shape
        luminance = (0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3])
        avg_luminance = self.global_pool(luminance)
        avg_luminance = avg_luminance.view(b, 1)
        
        channel_avg = img.mean(dim=[2, 3])
        
        exposure_level = torch.tanh(self.exposure_predictor(channel_avg))
        direction_logits = self.direction_predictor(channel_avg)
        exposure_direction = torch.softmax(direction_logits, dim=1)
        
        return exposure_level, exposure_direction, avg_luminance


class AdaptiveGammaCorrection(nn.Module):
    """Adaptive gamma correction based on exposure level"""
    def __init__(self):
        super(AdaptiveGammaCorrection, self).__init__()
        self.gamma_base = nn.Parameter(torch.ones(1), requires_grad=True)
        
    def forward(self, img, exposure_level, exposure_direction):
        """Apply adaptive gamma correction"""
        b = img.shape[0]
        
        adaptive_gamma = 1.0 + 0.6 * exposure_level.squeeze(-1)
        adaptive_gamma = torch.clamp(adaptive_gamma, min=0.3, max=2.5)
        
        img_corrected = torch.pow(torch.clamp(img, min=1e-8), 1.0 / adaptive_gamma.view(b, 1, 1, 1))
        
        return img_corrected, adaptive_gamma.unsqueeze(-1)


class WhiteBalanceCorrection(nn.Module):
    """White balance correction with color temperature and saturation adjustment"""
    def __init__(self, dim=64):
        super(WhiteBalanceCorrection, self).__init__()
        self.dim = dim
        
        self.color_base = nn.Parameter(torch.eye(3), requires_grad=True)
        
        self.color_temp_predictor = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )
        
        self.channel_gains = nn.Parameter(torch.ones(3), requires_grad=True)
        
        self.saturation_predictor = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, 1),
        )

    def forward(self, img):
        """Apply white balance correction"""
        b, c, h, w = img.shape
        channel_avg = img.mean(dim=[2, 3])
        
        color_temp = torch.tanh(self.color_temp_predictor(channel_avg))
        saturation = torch.tanh(self.saturation_predictor(channel_avg))
        
        adaptive_gains = self.channel_gains.clone()
        
        if color_temp is not None:
            temp_value = color_temp.squeeze(-1)
            temp_strength = 0.3
            
            r_scale = 1.0 - temp_value * temp_strength
            g_scale = 1.0 + torch.abs(temp_value) * 0.1
            b_scale = 1.0 + temp_value * temp_strength
            
            adaptive_gains = torch.stack([r_scale, g_scale, b_scale], dim=1)
        
        img_wb = img * adaptive_gains.view(b, 3, 1, 1)
        
        img_flat = img_wb.view(b, 3, -1).permute(0, 2, 1)
        img_color = torch.bmm(img_flat, self.color_base.unsqueeze(0).expand(b, -1, -1))
        img_color = img_color.permute(0, 2, 1).view(b, 3, h, w)
        img_color = torch.clamp(img_color, 0, 1)
        
        wb_params = {
            'color_temp': color_temp,
            'saturation': saturation,
            'channel_gains': adaptive_gains
        }
        
        return img_color, self.color_base, wb_params


class ExposureWhiteBalanceModule(nn.Module):
    """Combined exposure and white balance correction module"""
    def __init__(self, dim=64):
        super(ExposureWhiteBalanceModule, self).__init__()
        self.exposure_analyzer = ExposureAnalyzer(dim=dim)
        self.adaptive_gamma = AdaptiveGammaCorrection()
        self.white_balance = WhiteBalanceCorrection(dim=dim)
        
    def forward(self, img):
        """Apply exposure analysis, gamma correction, and white balance"""
        exposure_level, exposure_direction, luminance = self.exposure_analyzer(img)
        img_gamma, gamma_applied = self.adaptive_gamma(img, exposure_level, exposure_direction)
        img_wb, color_matrix, wb_params = self.white_balance(img_gamma)
        
        outputs = {
            'output': img_wb,
            'exposure_level': exposure_level,
            'exposure_direction': exposure_direction,
            'luminance': luminance,
            'gamma_applied': gamma_applied,
            'color_matrix': color_matrix,
            'wb_params': wb_params,
            'gamma_intermediate': img_gamma,
        }
        
        return outputs


if __name__ == "__main__":
    module = ExposureWhiteBalanceModule(dim=64)
    test_img = torch.randn(2, 3, 256, 256).clamp(0, 1)
    outputs = module(test_img)
    
    print(f"Input: {test_img.shape} | Output: {outputs['output'].shape}")
    print(f"Exposure level: {outputs['exposure_level'].shape}")
    print(f"Gamma applied: {outputs['gamma_applied'].shape}")

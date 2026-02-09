"""
Enhanced IAT Model with Exposure and White Balance modules
This version integrates exposure analysis and white balance correction
"""
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
import os
import math

from timm.models.layers import trunc_normal_
from model.blocks import CBlock_ln, SwinTransformerBlock
from model.global_net import Global_pred
from model.exposure_module import ExposureWhiteBalanceModule


class Local_pred(nn.Module):
    def __init__(self, dim=16, number=4, type='ccc'):
        super(Local_pred, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(3, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)  # head number
        if type =='ccc':  
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type =='ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type =='cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        self.mul_blocks = nn.Sequential(*blocks1, nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_blocks = nn.Sequential(*blocks2, nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())

    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        mul = self.mul_blocks(img1)
        add = self.add_blocks(img1)
        return mul, add


class Local_pred_S(nn.Module):
    """Short Cut Connection on Final Layer"""
    def __init__(self, in_dim=3, dim=16, number=4, type='ccc'):
        super(Local_pred_S, self).__init__()
        # initial convolution
        self.conv1 = nn.Conv2d(in_dim, dim, 3, padding=1, groups=1)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        # main blocks
        block = CBlock_ln(dim)
        block_t = SwinTransformerBlock(dim)
        if type =='ccc':
            blocks1 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
            blocks2 = [CBlock_ln(16, drop_path=0.01), CBlock_ln(16, drop_path=0.05), CBlock_ln(16, drop_path=0.1)]
        elif type =='ttt':
            blocks1, blocks2 = [block_t for _ in range(number)], [block_t for _ in range(number)]
        elif type =='cct':
            blocks1, blocks2 = [block, block, block_t], [block, block, block_t]
        self.mul_blocks = nn.Sequential(*blocks1)
        self.add_blocks = nn.Sequential(*blocks2)

        self.mul_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.ReLU())
        self.add_end = nn.Sequential(nn.Conv2d(dim, 3, 3, 1, 1), nn.Tanh())
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()
            
    def forward(self, img):
        img1 = self.relu(self.conv1(img))
        # short cut connection
        mul = self.mul_blocks(img1) + img1
        add = self.add_blocks(img1) + img1
        mul = self.mul_end(mul)
        add = self.add_end(add)
        return mul, add


class IAT(nn.Module):
    """Original IAT model - backward compatible"""
    def __init__(self, in_dim=3, with_global=True, type='lol'):
        super(IAT, self).__init__()
        self.local_net = Local_pred_S(in_dim=in_dim)
        self.with_global = with_global
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low):
        mul, add = self.local_net(img_low)
        img_high = (img_low.mul(mul)).add(add)

        if not self.with_global:
            return mul, add, img_high
        else:
            gamma, color = self.global_net(img_low)
            b = img_high.shape[0]
            img_high = img_high.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            img_high = torch.stack([self.apply_color(img_high[i,:,:,:], color[i,:,:])**gamma[i,:] for i in range(b)], dim=0)
            img_high = img_high.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)
            return mul, add, img_high


class IAT_Enhanced(nn.Module):
    """
    Enhanced IAT with dedicated Exposure and White Balance modules
    
    Pipeline:
    1. Input image -> Local enhancement (mul, add streams)
    2. Local enhanced image -> Exposure & White Balance correction
    3. Output corrected enhanced image
    
    Compared to original IAT:
    - Better handling of underexposure and overexposure
    - More robust white balance through dedicated color temperature module
    - Explicit exposure analysis guides the correction
    """
    def __init__(self, in_dim=3, with_global=True, with_exposure_wb=True, type='lol'):
        super(IAT_Enhanced, self).__init__()
        self.local_net = Local_pred_S(in_dim=in_dim)
        self.with_global = with_global
        self.with_exposure_wb = with_exposure_wb
        
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)
            
        if self.with_exposure_wb:
            self.exposure_wb_module = ExposureWhiteBalanceModule(dim=64)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low, return_intermediate=False):
        """
        Args:
            img_low: (B, 3, H, W) input image
            return_intermediate: if True, return intermediate results for analysis/loss computation
            
        Returns:
            if return_intermediate:
                dict with all intermediate results
            else:
                mul, add, img_high (for backward compatibility)
        """
        # Step 1: Local enhancement
        mul, add = self.local_net(img_low)
        img_enhanced = (img_low.mul(mul)).add(add)

        # Step 2: Global adjustment (gamma + color)
        if self.with_global:
            gamma, color = self.global_net(img_low)
            b = img_enhanced.shape[0]
            img_enhanced_perm = img_enhanced.permute(0, 2, 3, 1)  # (B,C,H,W) -- (B,H,W,C)
            img_enhanced = torch.stack([self.apply_color(img_enhanced_perm[i,:,:,:], color[i,:,:])**gamma[i,:] for i in range(b)], dim=0)
            img_enhanced = img_enhanced.permute(0, 3, 1, 2)  # (B,H,W,C) -- (B,C,H,W)

        # Step 3: Exposure and white balance correction
        if self.with_exposure_wb:
            exposure_wb_outputs = self.exposure_wb_module(img_enhanced)
            img_high = exposure_wb_outputs['output']
            
            if return_intermediate:
                return {
                    'output': img_high,
                    'mul': mul,
                    'add': add,
                    'img_before_exposure_wb': img_enhanced,
                    'local_enhanced': img_enhanced if not self.with_global else img_enhanced,
                    'exposure_wb': exposure_wb_outputs,
                    'gamma': gamma if self.with_global else None,
                    'color': color if self.with_global else None,
                }
        else:
            img_high = img_enhanced
            
        return mul, add, img_high


class IAT_Advanced(nn.Module):
    """
    Advanced IAT with exposure-aware local enhancement
    
    The local network is aware of exposure level and can adapt its behavior
    - More aggressive enhancement for underexposed regions
    - More careful enhancement for overexposed regions to avoid blown highlights
    """
    def __init__(self, in_dim=3, with_global=True, with_exposure_wb=True, type='lol'):
        super(IAT_Advanced, self).__init__()
        
        # Exposure analyzer (shared)
        self.exposure_analyzer = ExposureWhiteBalanceModule(dim=64)
        
        # Main local enhancement network
        self.local_net = Local_pred_S(in_dim=in_dim)
        
        # Exposure-aware local enhancement (additional channel)
        # This learns adjustments specific to exposure conditions
        self.exposure_aware_net = Local_pred_S(in_dim=in_dim, dim=16)
        
        self.with_global = with_global
        self.with_exposure_wb = with_exposure_wb
        
        if self.with_global:
            self.global_net = Global_pred(in_channels=in_dim, type=type)

    def apply_color(self, image, ccm):
        shape = image.shape
        image = image.view(-1, 3)
        image = torch.tensordot(image, ccm, dims=[[-1], [-1]])
        image = image.view(shape)
        return torch.clamp(image, 1e-8, 1.0)

    def forward(self, img_low, return_intermediate=False):
        """
        Args:
            img_low: (B, 3, H, W) input image
            return_intermediate: if True, return intermediate results
        """
        # Step 0: Analyze exposure level
        exposure_outputs = self.exposure_analyzer(img_low)
        exposure_level = exposure_outputs['exposure_level']  # (B, 1)
        
        # Step 1: Base local enhancement
        mul_base, add_base = self.local_net(img_low)
        
        # Step 2: Exposure-aware enhancement
        mul_exp, add_exp = self.exposure_aware_net(img_low)
        
        # Step 3: Adaptively blend based on exposure level
        # exposure_level: -1 (dark) to +1 (bright)
        # For dark images: use more aggressive enhancement from exposure-aware path
        # For bright images: use more conservative enhancement
        blend_weight = torch.sigmoid(exposure_level)  # (B, 1) -> range [0, 1]
        blend_weight = blend_weight.view(blend_weight.shape[0], blend_weight.shape[1], 1, 1)  # (B, 1, 1, 1)
        
        mul = mul_base * blend_weight + mul_exp * (1 - blend_weight)
        add = add_base * blend_weight + add_exp * (1 - blend_weight)
        
        img_enhanced = (img_low.mul(mul)).add(add)

        # Step 4: Global adjustment
        if self.with_global:
            gamma, color = self.global_net(img_low)
            b = img_enhanced.shape[0]
            img_enhanced_perm = img_enhanced.permute(0, 2, 3, 1)
            img_enhanced = torch.stack([self.apply_color(img_enhanced_perm[i,:,:,:], color[i,:,:])**gamma[i,:] for i in range(b)], dim=0)
            img_enhanced = img_enhanced.permute(0, 3, 1, 2)

        # Step 5: Final exposure and white balance correction
        if self.with_exposure_wb:
            exposure_wb_outputs = self.exposure_analyzer(img_enhanced)
            img_high = exposure_wb_outputs['output']
            
            if return_intermediate:
                return {
                    'output': img_high,
                    'mul': mul,
                    'add': add,
                    'mul_base': mul_base,
                    'mul_exp': mul_exp,
                    'add_base': add_base,
                    'add_exp': add_exp,
                    'blend_weight': blend_weight,
                    'exposure_level_input': exposure_level,
                    'exposure_wb': exposure_wb_outputs,
                }
        else:
            img_high = img_enhanced
            
        return mul, add, img_high


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES']='3'
    
    # Test original IAT
    print("=" * 50)
    print("Testing Original IAT")
    print("=" * 50)
    img = torch.Tensor(1, 3, 256, 384).clamp(0, 1)
    net = IAT()
    print('Total parameters (IAT):', sum(param.numel() for param in net.parameters()))
    _, _, high = net(img)
    print(f'Input shape: {img.shape}, Output shape: {high.shape}')
    
    # Test enhanced IAT
    print("\n" + "=" * 50)
    print("Testing Enhanced IAT")
    print("=" * 50)
    net_enhanced = IAT_Enhanced()
    print('Total parameters (IAT_Enhanced):', sum(param.numel() for param in net_enhanced.parameters()))
    _, _, high = net_enhanced(img)
    print(f'Input shape: {img.shape}, Output shape: {high.shape}')
    
    # Test advanced IAT
    print("\n" + "=" * 50)
    print("Testing Advanced IAT")
    print("=" * 50)
    net_advanced = IAT_Advanced()
    print('Total parameters (IAT_Advanced):', sum(param.numel() for param in net_advanced.parameters()))
    _, _, high = net_advanced(img)
    print(f'Input shape: {img.shape}, Output shape: {high.shape}')
    
    # Test with intermediate outputs
    print("\n" + "=" * 50)
    print("Testing with intermediate outputs")
    print("=" * 50)
    outputs = net_advanced(img, return_intermediate=True)
    print("Intermediate keys:", list(outputs.keys()))

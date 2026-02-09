"""
Multi-Exposure Fusion Encoder
Processes one or multiple exposure images and fuses them into unified feature map

Features:
- Residual Dense Block feature extraction
- Multi-scale fusion
- Cross-attention mechanism
- Channel + Spatial attention
- Learnable fusion weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class ConvBlock(nn.Module):
    """Basic Conv Block: Conv + BN + LeakyReLU"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with dense connections"""
    def __init__(self, num_fea, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.num_fea = num_fea
        self.num_grow_ch = num_grow_ch
        
        self.conv1 = nn.Conv2d(num_fea, num_grow_ch, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_fea + num_grow_ch, num_grow_ch, 3, 1, 1, bias=True)
        self.conv3 = nn.Conv2d(num_fea + 2 * num_grow_ch, num_grow_ch, 3, 1, 1, bias=True)
        self.conv4 = nn.Conv2d(num_fea + 3 * num_grow_ch, num_grow_ch, 3, 1, 1, bias=True)
        self.conv5 = nn.Conv2d(num_fea + 4 * num_grow_ch, num_fea, 3, 1, 1, bias=True)
        
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
    
    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 + x  # Residual connection


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    def __init__(self, num_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // reduction, num_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)"""
    def __init__(self, num_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(num_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


class SharedFeatureExtractor(nn.Module):
    """Powerful shared encoder with Residual Dense Blocks for feature extraction"""
    def __init__(self, in_channels=3, out_channels=64, num_rdb=3, num_grow_ch=32):
        super(SharedFeatureExtractor, self).__init__()
        self.out_channels = out_channels
        
        self.conv_first = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True)
        
        self.conv_s1 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=True)
        self.rdb_s1 = nn.Sequential(*[ResidualDenseBlock(out_channels, num_grow_ch) for _ in range(num_rdb)])
        self.attention_s1 = CBAM(out_channels)
        
        self.conv_s2 = nn.Conv2d(out_channels, out_channels, 3, stride=2, padding=1, bias=True)
        self.rdb_s2 = nn.Sequential(*[ResidualDenseBlock(out_channels, num_grow_ch) for _ in range(num_rdb)])
        self.attention_s2 = CBAM(out_channels)
        
        self.rdb_final = nn.Sequential(*[ResidualDenseBlock(out_channels, num_grow_ch) for _ in range(num_rdb)])
        self.attention_final = CBAM(out_channels)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        """Extract multi-scale features"""
        fea = self.lrelu(self.conv_first(x))
        
        feat_s1 = self.lrelu(self.conv_s1(fea))
        feat_s1 = self.rdb_s1(feat_s1)
        feat_s1 = self.attention_s1(feat_s1)
        
        feat_s2 = self.lrelu(self.conv_s2(feat_s1))
        feat_s2 = self.rdb_s2(feat_s2)
        feat_s2 = self.attention_s2(feat_s2)
        
        feat_final = self.rdb_final(feat_s2)
        feat_final = self.attention_final(feat_final)
        
        return {
            's1': feat_s1,
            's2': feat_s2,
            'final': feat_final
        }


class CrossAttentionFusion(nn.Module):
    """Cross-Attention Fusion Module for multiple images"""
    def __init__(self, num_channels=64, num_heads=4):
        super(CrossAttentionFusion, self).__init__()
        self.num_channels = num_channels
        self.num_heads = num_heads
        self.head_dim = num_channels // num_heads
        
        assert num_channels % num_heads == 0
        
        self.query = nn.Conv2d(num_channels, num_channels, 1)
        self.key = nn.Conv2d(num_channels, num_channels, 1)
        self.value = nn.Conv2d(num_channels, num_channels, 1)
        
        self.proj = nn.Conv2d(num_channels, num_channels, 1)
        self.fusion_weight = nn.Parameter(torch.ones(1))
        
        self.scale = self.head_dim ** -0.5
    
    def forward(self, features_list):
        """Apply cross-attention fusion to multiple feature maps"""
        B, C, H, W = features_list[0].shape
        N = len(features_list)
        
        all_features = torch.stack(features_list, dim=1)
        all_features_flat = all_features.reshape(B*N, C, H*W)
        
        query = self.query(all_features_flat.reshape(B*N, C, H, W)).reshape(B*N, C, H*W)
        key = self.key(all_features_flat.reshape(B*N, C, H, W)).reshape(B*N, C, H*W)
        value = self.value(all_features_flat.reshape(B*N, C, H, W)).reshape(B*N, C, H*W)
        
        query = query.reshape(B*N, self.num_heads, self.head_dim, H*W)
        key = key.reshape(B*N, self.num_heads, self.head_dim, H*W)
        value = value.reshape(B*N, self.num_heads, self.head_dim, H*W)
        
        attn = torch.matmul(query.transpose(2, 3), key)
        attn = attn * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, value.transpose(2, 3))
        out = out.transpose(2, 3).reshape(B*N, C, H*W)
        
        out = self.proj(out.reshape(B*N, C, H, W))
        out = out.reshape(B, N, C, H, W)
        
        fused = torch.mean(out, dim=1)
        
        return fused


class AdaptiveWeightedFusion(nn.Module):
    """Adaptive Weighted Fusion with learnable per-image weights"""
    def __init__(self, num_channels=64):
        super(AdaptiveWeightedFusion, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        self.importance_net = nn.Sequential(
            nn.Conv2d(num_channels, num_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_channels // 4, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, features_list):
        """Compute importance scores and perform weighted fusion"""
        importance_scores = []
        for feat in features_list:
            score = self.importance_net(feat)
            importance_scores.append(score)
        
        importance_scores = torch.cat(importance_scores, dim=1)
        importance_scores = F.softmax(importance_scores, dim=1)
        
        fused = torch.zeros_like(features_list[0])
        for i, feat in enumerate(features_list):
            weight = importance_scores[:, i:i+1, :, :]
            fused = fused + weight * feat
        
        return fused


class MultiScaleFusion(nn.Module):
    """Multi-scale feature fusion combining multiple scales"""
    def __init__(self, num_channels=64):
        super(MultiScaleFusion, self).__init__()
        
        self.fusion_s1_to_s2 = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        self.fusion_s2_refined = nn.Sequential(
            ResidualDenseBlock(num_channels * 2, 32),
            nn.Conv2d(num_channels * 2, num_channels, 1)
        )
    
    def forward(self, feat_s1, feat_s2):
        """Fuse scale s1 and s2 features"""
        feat_s1_down = self.fusion_s1_to_s2(feat_s1)
        fused = torch.cat([feat_s2, feat_s1_down], dim=1)
        fused = self.fusion_s2_refined(fused)
        
        return fused


class FeatureRefinement(nn.Module):
    """Feature refinement with residual dense blocks and attention"""
    def __init__(self, num_channels=64, num_rdb=2):
        super(FeatureRefinement, self).__init__()
        
        self.refine = nn.Sequential(
            *[ResidualDenseBlock(num_channels, 32) for _ in range(num_rdb)],
            CBAM(num_channels)
        )
    
    def forward(self, x):
        return self.refine(x)


class MultiExposureEncoder(nn.Module):
    """Advanced Multi-Exposure Fusion Encoder
    
    Processes one or multiple exposure images:
    - Extract powerful features via Residual Dense Blocks
    - Multi-scale processing
    - Cross-attention fusion
    - Double attention (Channel + Spatial)
    - Learnable weights
    
    Input: List of N images or single image, each (B, 3, H, W)
    Output: Fused feature (B, 64, H//4, W//4)
    """
    
    def __init__(
        self,
        in_channels=3,
        out_channels=64,
        num_rdb=3,
        num_grow_ch=32,
        fusion_type='cross_attention',  # 'cross_attention' hoặc 'weighted'
        num_heads=4
    ):
        super(MultiExposureEncoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fusion_type = fusion_type
        
        print(f"[Encoder] Using {fusion_type} fusion")
        
        # Shared Feature Extractor (mạnh = RDB blocks)
        self.feature_extractor = SharedFeatureExtractor(
            in_channels=in_channels,
            out_channels=out_channels,
            num_rdb=num_rdb,
            num_grow_ch=num_grow_ch
        )
        
        # Multi-scale fusion
        self.multiscale_fusion = MultiScaleFusion(num_channels=out_channels)
        
        # Choose fusion strategy
        if fusion_type == 'cross_attention':
            self.fusion_module = CrossAttentionFusion(num_channels=out_channels, num_heads=num_heads)
        else:  # 'weighted'
            self.fusion_module = AdaptiveWeightedFusion(num_channels=out_channels)
        
        # Refinement
        self.refinement = FeatureRefinement(num_channels=out_channels, num_rdb=2)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, images):
        """Process images through multi-scale encoder and fusion"""
        if isinstance(images, torch.Tensor):
            images = [images]
        
        all_features = []
        for img in images:
            multi_scale_feats = self.feature_extractor(img)
            all_features.append(multi_scale_feats)
        
        feats_s1 = [f['s1'] for f in all_features]
        feats_s2 = [f['s2'] for f in all_features]
        feats_final = [f['final'] for f in all_features]
        
        if len(images) == 1:
            fused_s1 = feats_s1[0]
            fused_s2 = feats_s2[0]
            fused_final = feats_final[0]
        else:
            fused_s1 = self.fusion_module(feats_s1)
            fused_s2 = self.fusion_module(feats_s2)
            fused_final = self.fusion_module(feats_final)
        
        fused_s1s2 = self.multiscale_fusion(fused_s1, fused_s2)
        fused = fused_s1s2 + fused_final
        refined = self.refinement(fused)
        
        return refined


class MultiExposureEncoderWithVisualization(MultiExposureEncoder):
    """Extended encoder with visualization capabilities"""
    
    def forward(self, images, return_weights=False):
        """Process with optional fusion information"""
        if isinstance(images, torch.Tensor):
            images = [images]
        
        all_features = []
        for img in images:
            multi_scale_feats = self.feature_extractor(img)
            all_features.append(multi_scale_feats)
        
        feats_final = [f['final'] for f in all_features]
        
        if len(images) == 1:
            fused_final = feats_final[0]
        else:
            fused_final = self.fusion_module(feats_final)
        
        refined = self.refinement(fused_final)
        
        if return_weights:
            return refined, {
                'num_images': len(images),
                'fusion_type': self.fusion_type
            }
        else:
            return refined


if __name__ == "__main__":
    print("=" * 70)
    print("Testing ADVANCED MultiExposureEncoder")
    print("=" * 70)
    
    print("\n[Test 1] Single image + Cross-Attention Fusion")
    encoder = MultiExposureEncoder(
        in_channels=3,
        out_channels=64,
        num_rdb=3,
        fusion_type='cross_attention'
    )
    img = torch.randn(2, 3, 1200, 900)
    feat = encoder([img])
    print(f"✓ Input: 1 image {img.shape}")
    print(f"✓ Output: {feat.shape}")
    assert feat.shape == (2, 64, 300, 225)
    
    print("\n[Test 2] Multiple images (N=3) + Cross-Attention")
    imgs = [torch.randn(2, 3, 1200, 900) for _ in range(3)]
    feat = encoder(imgs)
    print(f"✓ Input: {len(imgs)} images, each {imgs[0].shape}")
    print(f"✓ Output: {feat.shape}")
    assert feat.shape == (2, 64, 300, 225)
    
    print("\n[Test 3] Multiple images + Weighted Fusion")
    encoder_weighted = MultiExposureEncoder(
        in_channels=3,
        out_channels=64,
        num_rdb=2,
        fusion_type='weighted'
    )
    imgs = [torch.randn(1, 3, 1200, 900) for _ in range(3)]
    feat = encoder_weighted(imgs)
    print(f"✓ Input: {len(imgs)} images, each {imgs[0].shape}")
    print(f"✓ Output: {feat.shape}")
    assert feat.shape == (1, 64, 300, 225)
    
    print("\n[Test 4] With weight visualization")
    encoder_vis = MultiExposureEncoderWithVisualization(
        in_channels=3,
        out_channels=64,
        fusion_type='cross_attention'
    )
    imgs = [torch.randn(1, 3, 1200, 900) for _ in range(3)]
    feat, info = encoder_vis(imgs, return_weights=True)
    print(f"✓ Feature shape: {feat.shape}")
    print(f"✓ Fusion info: {info}")
    
    print("\n[Test 5] Model Complexity")
    total_params = sum(p.numel() for p in encoder.parameters())
    trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
    print(f"✓ Total parameters: {total_params:,}")
    print(f"✓ Trainable parameters: {trainable_params:,}")
    
    print("\n[Test 6] Memory Efficiency")
    batch_size = 2
    imgs = [torch.randn(batch_size, 3, 1200, 900) for _ in range(3)]
    
    encoder.eval()
    with torch.no_grad():
        feat = encoder(imgs)
    
    input_size_mb = sum(img.element_size() * img.nelement() / (1024**2) for img in imgs)
    output_size_mb = feat.element_size() * feat.nelement() / (1024**2)
    
    print(f"✓ Input size: {input_size_mb:.2f} MB (3 images × {imgs[0].shape})")
    print(f"✓ Output size: {output_size_mb:.2f} MB ({feat.shape})")
    print(f"✓ Compression: {input_size_mb/output_size_mb:.1f}x")
    
    print("\n" + "=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)

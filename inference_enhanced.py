"""
Inference and Evaluation script for Enhanced IAT
Demonstrates how to use the model for inference with analysis
"""

import os
import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from pathlib import Path
import csv
from datetime import datetime

from model.IAT_main import IAT_Enhanced, IAT_Advanced
from model.multi_exposure_encoder import MultiExposureEncoder
from model.losses import ExposureWhiteBalanceLoss


class EnhancementAnalyzer:
    """Analyze enhancement results"""
    
    @staticmethod
    def calculate_luminance(img):
        """Calculate luminance (Y = 0.299*R + 0.587*G + 0.114*B)"""
        if img.dim() == 4:
            lum = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
        else:
            lum = 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]
        return lum
    
    @staticmethod
    def exposure_analysis(img_before, img_after):
        """Analyze exposure correction"""
        lum_before = EnhancementAnalyzer.calculate_luminance(img_before)
        lum_after = EnhancementAnalyzer.calculate_luminance(img_after)
        
        analysis = {
            'mean_luminance_before': lum_before.mean().item(),
            'mean_luminance_after': lum_after.mean().item(),
            'std_luminance_before': lum_before.std().item(),
            'std_luminance_after': lum_after.std().item(),
            'exposure_change': (lum_after.mean() - lum_before.mean()).item(),
        }
        
        before_min, before_max = lum_before.min().item(), lum_before.max().item()
        after_min, after_max = lum_after.min().item(), lum_after.max().item()
        
        analysis['dynamic_range_before'] = before_max - before_min
        analysis['dynamic_range_after'] = after_max - after_min
        
        return analysis
    
    @staticmethod
    def white_balance_analysis(img_before, img_after):
        """Analyze white balance correction"""
        if img_before.dim() == 4:
            r_before = img_before[:, 0].mean()
            g_before = img_before[:, 1].mean()
            b_before = img_before[:, 2].mean()
            
            r_after = img_after[:, 0].mean()
            g_after = img_after[:, 1].mean()
            b_after = img_after[:, 2].mean()
        else:
            r_before = img_before[0].mean()
            g_before = img_before[1].mean()
            b_before = img_before[2].mean()
            
            r_after = img_after[0].mean()
            g_after = img_after[1].mean()
            b_after = img_after[2].mean()
        
        wb_before = torch.abs(r_before - g_before) + torch.abs(g_before - b_before)
        wb_after = torch.abs(r_after - g_after) + torch.abs(g_after - b_after)
        
        analysis = {
            'wb_before': wb_before.item(),
            'wb_after': wb_after.item(),
            'wb_improvement': (wb_before - wb_after).item(),
            'r_mean_before': r_before.item(),
            'g_mean_before': g_before.item(),
            'b_mean_before': b_before.item(),
            'r_mean_after': r_after.item(),
            'g_mean_after': g_after.item(),
            'b_mean_after': b_after.item(),
        }
        
        return analysis
    
    @staticmethod
    def contrast_analysis(img_before, img_after):
        """Analyze contrast improvement"""
        lum_before = EnhancementAnalyzer.calculate_luminance(img_before)
        lum_after = EnhancementAnalyzer.calculate_luminance(img_after)
        
        contrast_before = lum_before.std().item()
        contrast_after = lum_after.std().item()
        
        return {
            'contrast_before': contrast_before,
            'contrast_after': contrast_after,
            'contrast_improvement': contrast_after - contrast_before,
        }


class EnhancementInference:
    """Inference pipeline for enhancement with encoder support"""
    
    def __init__(self, model_path, model_type='IAT_Enhanced', device='cuda', use_encoder=True):
        self.device = torch.device(device)
        self.model_type = model_type
        self.use_encoder = use_encoder
        
        # Load model and encoder
        self._load_model(model_path)
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        self.analyzer = EnhancementAnalyzer()
    
    def _load_model(self, model_path):
        """Load model and encoder from checkpoint"""
        self.encoder = None
        if self.use_encoder:
            print("[Encoder] Initializing Multi-Exposure Encoder...")
            self.encoder = MultiExposureEncoder(
                in_channels=3,
                out_channels=64,
                num_rdb=3,
                num_grow_ch=32,
                fusion_type='cross_attention',
                num_heads=4
            )
            self.encoder.to(self.device)
            self.encoder.eval()
            print("[Encoder] ✓ Initialized")
        
        in_dim = 64 if self.use_encoder else 3
        
        if self.model_type == 'IAT_Enhanced':
            self.model = IAT_Enhanced(
                in_dim=in_dim,
                with_global=True,
                with_exposure_wb=True
            )
        elif self.model_type == 'IAT_Advanced':
            self.model = IAT_Advanced(
                in_dim=in_dim,
                with_global=True,
                with_exposure_wb=True
            )
        
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            
            if isinstance(checkpoint, dict):
                if 'encoder_state_dict' in checkpoint and self.encoder is not None:
                    self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
                    print(f"[Encoder] State loaded from checkpoint")
                
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
            else:
                self.model.load_state_dict(checkpoint)
            
            print(f"[Model] ✓ Loaded from {model_path}")
        else:
            print(f"[Warning] Model path {model_path} not found, using random initialized model")
        
        self.model.to(self.device)
        self.model.eval()
    
    def enhance_image(self, img_path, return_analysis=True, return_intermediate=False):
        """Enhance single image with optional analysis and intermediate outputs"""
        img = Image.open(img_path).convert('RGB')
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        img_tensor = torch.clamp(img_tensor, 0, 1)
        img_original = img_tensor.clone()
        
        with torch.no_grad():
            if self.use_encoder and self.encoder is not None:
                fused_features = self.encoder([img_tensor])
                
                if return_intermediate:
                    outputs = self.model(fused_features, return_intermediate=True)
                    img_enhanced = outputs['output']
                    intermediate = outputs
                else:
                    mul, add, img_enhanced = self.model(fused_features)
                    intermediate = None
            else:
                if return_intermediate:
                    outputs = self.model(img_tensor, return_intermediate=True)
                    img_enhanced = outputs['output']
                    intermediate = outputs
                else:
                    mul, add, img_enhanced = self.model(img_tensor)
                    intermediate = None
        
        img_enhanced = torch.clamp(img_enhanced, 0, 1)
        
        analysis = {}
        if return_analysis:
            exposure_analysis = self.analyzer.exposure_analysis(img_original, img_enhanced)
            wb_analysis = self.analyzer.white_balance_analysis(img_original, img_enhanced)
            contrast_analysis = self.analyzer.contrast_analysis(img_original, img_enhanced)
            analysis = {**exposure_analysis, **wb_analysis, **contrast_analysis}
        
        return img_enhanced, analysis, intermediate
    
    def enhance_multi_exposure(self, img_paths_list, return_analysis=True):
        """Enhance using multiple exposure images"""
        if not self.use_encoder or self.encoder is None:
            raise ValueError("Encoder must be enabled for multi-exposure inference")
        
        if len(img_paths_list) < 1:
            raise ValueError("At least 1 image required")
        
        img_tensors = []
        for img_path in img_paths_list:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.transform(img).unsqueeze(0).to(self.device)
            img_tensor = torch.clamp(img_tensor, 0, 1)
            img_tensors.append(img_tensor)
        
        img_reference = img_tensors[0].clone()
        
        with torch.no_grad():
            fused_features = self.encoder(img_tensors)
            mul, add, img_enhanced = self.model(fused_features)
        
        img_enhanced = torch.clamp(img_enhanced, 0, 1)
        
        analysis = {}
        if return_analysis:
            exposure_analysis = self.analyzer.exposure_analysis(img_reference, img_enhanced)
            wb_analysis = self.analyzer.white_balance_analysis(img_reference, img_enhanced)
            contrast_analysis = self.analyzer.contrast_analysis(img_reference, img_enhanced)
            analysis = {
                'num_exposures': len(img_paths_list),
                **exposure_analysis, **wb_analysis, **contrast_analysis
            }
        
        return img_enhanced, analysis
    
    def enhance_batch(self, img_dir, output_dir, save_results=True):
        """Enhance batch of images"""
        img_dir = Path(img_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = output_dir / 'enhancement_results.csv'
        csv_file = open(results_file, 'w', newline='')
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=['image', 'exposure_change', 'luminance_before', 'luminance_after',
                       'wb_improvement', 'contrast_improvement']
        )
        csv_writer.writeheader()
        
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(img_dir.glob(f'**/{ext}'))
            image_files.extend(img_dir.glob(f'**/{ext.upper()}'))
        
        print(f"Found {len(image_files)} images")
        
        for img_path in image_files:
            try:
                img_enhanced, analysis, _ = self.enhance_image(str(img_path))
                
                if save_results:
                    enhanced_pil = transforms.ToPILImage()(img_enhanced[0].cpu())
                    output_path = output_dir / img_path.relative_to(img_dir).parent / f"{img_path.stem}_enhanced.jpg"
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    enhanced_pil.save(str(output_path), quality=95)
                
                csv_writer.writerow({
                    'image': str(img_path.relative_to(img_dir)),
                    'exposure_change': f"{analysis.get('exposure_change', 0):.4f}",
                    'luminance_before': f"{analysis.get('mean_luminance_before', 0):.4f}",
                    'luminance_after': f"{analysis.get('mean_luminance_after', 0):.4f}",
                    'wb_improvement': f"{analysis.get('wb_improvement', 0):.4f}",
                    'contrast_improvement': f"{analysis.get('contrast_improvement', 0):.4f}",
                })
                csv_file.flush()
                
                print(f"✓ Enhanced: {img_path.name}")
                
            except Exception as e:
                print(f"✗ Failed: {img_path.name} - {str(e)}")
        
        csv_file.close()
        print(f"Results saved to {results_file}")


def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced IAT Inference')
    
    parser.add_argument('--mode', type=str, default='single', 
                       choices=['single', 'multi_exposure', 'batch'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--model_type', type=str, default='IAT_Enhanced',
                       choices=['IAT_Enhanced', 'IAT_Advanced'])
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--use_encoder', type=bool, default=True)
    parser.add_argument('--save_analysis', action='store_true')
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Initialize inference
    inference = EnhancementInference(
        model_path=args.model_path,
        model_type=args.model_type,
        device=args.device
    )
    
    if args.mode == 'single':
        print(f"Enhancing image: {args.input_path}")
        img_enhanced, analysis, intermediate = inference.enhance_image(
            args.input_path,
            return_analysis=True,
            return_intermediate=True
        )
        
        enhanced_pil = transforms.ToPILImage()(img_enhanced[0].cpu())
        enhanced_pil.save(args.output_path, quality=95)
        print(f"✓ Saved to {args.output_path}")
        
        print("\n=== Enhancement Analysis ===")
        print(f"Exposure Change: {analysis['exposure_change']:.4f}")
        print(f"Luminance Before: {analysis['mean_luminance_before']:.4f}")
        print(f"Luminance After: {analysis['mean_luminance_after']:.4f}")
        print(f"Dynamic Range Before: {analysis['dynamic_range_before']:.4f}")
        print(f"Dynamic Range After: {analysis['dynamic_range_after']:.4f}")
        print(f"WB Improvement: {analysis['wb_improvement']:.4f}")
        print(f"Contrast Improvement: {analysis['contrast_improvement']:.4f}")
        
        if intermediate and 'exposure_wb' in intermediate:
            exp_wb = intermediate['exposure_wb']
            print(f"\nExposure Level: {exp_wb['exposure_level'].item():.4f}")
            print(f"Luminance: {exp_wb['luminance'].item():.4f}")
            if 'wb_params' in exp_wb and 'color_temp' in exp_wb['wb_params']:
                print(f"Color Temperature: {exp_wb['wb_params']['color_temp'].item():.4f}")
    
    elif args.mode == 'multi_exposure':
        img_paths = args.input_path.split(',')
        print(f"Enhancing {len(img_paths)} exposures...")
        print(f"Paths: {img_paths}")
        
        img_enhanced, analysis = inference.enhance_multi_exposure(img_paths, return_analysis=True)
        
        enhanced_pil = transforms.ToPILImage()(img_enhanced[0].cpu())
        enhanced_pil.save(args.output_path, quality=95)
        print(f"✓ Saved to {args.output_path}")
        
        print("\n=== Multi-Exposure Fusion Analysis ===")
        print(f"Number of Exposures: {analysis['num_exposures']}")
        print(f"Exposure Change: {analysis['exposure_change']:.4f}")
        print(f"Luminance Before: {analysis['mean_luminance_before']:.4f}")
        print(f"Luminance After: {analysis['mean_luminance_after']:.4f}")
        print(f"Dynamic Range Before: {analysis['dynamic_range_before']:.4f}")
        print(f"Dynamic Range After: {analysis['dynamic_range_after']:.4f}")
        print(f"WB Improvement: {analysis['wb_improvement']:.4f}")
        print(f"Contrast Improvement: {analysis['contrast_improvement']:.4f}")
    
    elif args.mode == 'batch':
        print(f"Batch processing images from: {args.input_path}")
        inference.enhance_batch(args.input_path, args.output_path)


if __name__ == '__main__':
    main()

"""Training script for IAT model with multi-exposure fusion"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import logging
from tqdm import tqdm
from datetime import datetime
import math

from model.IAT_main import IAT_Enhanced, IAT_Advanced
from model.losses import ExposureWhiteBalanceLoss, CombinedLoss
from model.multi_exposure_encoder import MultiExposureEncoder
from data_loaders.multi_exposure import MultiExposureDynamicLoader


def collate_multi_exposure(batch):
    """Collate multi-exposure batch data"""
    exposures_by_idx = {}
    gts_batch = []
    
    for exposures_list, gt_tensor in batch:
        if isinstance(exposures_list, list) and len(exposures_list) > 0:
            for exp_idx, exp_tensor in enumerate(exposures_list):
                if exp_idx not in exposures_by_idx:
                    exposures_by_idx[exp_idx] = []
                exposures_by_idx[exp_idx].append(exp_tensor)
        gts_batch.append(gt_tensor)
    
    multi_exposure_list = []
    for exp_idx in sorted(exposures_by_idx.keys()):
        exp_batch = torch.stack(exposures_by_idx[exp_idx], dim=0)
        multi_exposure_list.append(exp_batch)
    
    gts_tensor = torch.stack(gts_batch, dim=0)
    
    return {
        'multi_exposure': multi_exposure_list,
        'label': gts_tensor
    }


def setup_logging(save_dir):
    log_file = os.path.join(save_dir, f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Enhanced IAT Training')
    
    parser.add_argument('--data_path', type=str, default='dataset')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--decay', type=float, default=0.5)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--model_type', type=str, default='IAT_Enhanced', 
                       choices=['IAT_Enhanced', 'IAT_Advanced'])
    parser.add_argument('--use_global', type=bool, default=True)
    parser.add_argument('--use_exposure_wb', type=bool, default=True)
    parser.add_argument('--variable_n', type=bool, default=True)
    parser.add_argument('--loss_type', type=str, default='ExposureWhiteBalance',
                       choices=['ExposureWhiteBalance', 'Combined'])
    parser.add_argument('--exposure_weight', type=float, default=1.0)
    parser.add_argument('--wb_weight', type=float, default=1.0)
    parser.add_argument('--contrast_weight', type=float, default=0.2)
    parser.add_argument('--color_const_weight', type=float, default=0.1)
    parser.add_argument('--photometric_weight', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--gpu_ids', type=str, default='0')
    
    return parser.parse_args()


class EnhancedIATTrainer:
    def __init__(self, args):
        self.args = args
        self.logger = setup_logging(args.save_dir)
        os.makedirs(args.save_dir, exist_ok=True)
        
        self.device = torch.device(args.device)
        self.logger.info(f"Using device: {self.device}")
        
        # Setup model
        self._setup_model()
        
        # Setup optimizer & scheduler
        self._setup_optimizer()
        
        # Setup loss function
        self._setup_loss()
        
        # Setup dataloader
        self._setup_dataloader()
        
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def _setup_model(self):
        """Initialize encoder and model"""
        self.logger.info(f"Initializing multi-exposure encoder...")
        
        self.encoder = MultiExposureEncoder(
            in_channels=3,
            out_channels=64,
            num_rdb=3,
            num_grow_ch=32,
            fusion_type='cross_attention',
            num_heads=4
        )
        self.encoder.to(self.device)
        
        encoder_out_dim = 64
        
        if self.args.model_type == 'IAT_Enhanced':
            self.model = IAT_Enhanced(
                in_dim=encoder_out_dim,
                with_global=self.args.use_global,
                with_exposure_wb=self.args.use_exposure_wb,
                type='lol'
            )
        elif self.args.model_type == 'IAT_Advanced':
            self.model = IAT_Advanced(
                in_dim=encoder_out_dim,
                with_global=self.args.use_global,
                with_exposure_wb=self.args.use_exposure_wb,
                type='lol'
            )
        
        self.model.to(self.device)
        
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        model_params = sum(p.numel() for p in self.model.parameters())
        total_params = encoder_params + model_params
        self.logger.info(f"Encoder: {encoder_params:,} | Model: {model_params:,} | Total: {total_params:,}")
        
        if self.args.resume_from:
            self._load_checkpoint(self.args.resume_from)
    
    def _setup_optimizer(self):
        """Setup optimizer and scheduler"""
        all_parameters = list(self.encoder.parameters()) + list(self.model.parameters())
        
        self.optimizer = torch.optim.Adam(
            all_parameters,
            lr=self.args.lr,
            betas=(0.9, 0.999),
            weight_decay=self.args.decay
        )
        
        def lr_lambda(current_step):
            if current_step < self.args.warmup_epochs:
                return float(current_step) / float(max(1, self.args.warmup_epochs))
            else:
                progress = float(current_step - self.args.warmup_epochs) / float(max(1, self.args.num_epochs - self.args.warmup_epochs))
                return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.logger.info("✓ Optimizer and scheduler initialized")
    
    def _setup_loss(self):
        """Setup loss function"""
        self.logger.info(f"Loss type: {self.args.loss_type}")
        
        if self.args.loss_type == 'ExposureWhiteBalance':
            self.criterion = ExposureWhiteBalanceLoss(
                exposure_weight=self.args.exposure_weight,
                wb_weight=self.args.wb_weight,
                contrast_weight=self.args.contrast_weight,
                color_constancy_weight=self.args.color_const_weight
            )
        elif self.args.loss_type == 'Combined':
            self.criterion = CombinedLoss(
                exposure_wb_weight=0.5,
                photometric_weight=self.args.photometric_weight,
                exposure_weight=self.args.exposure_weight,
                wb_weight=self.args.wb_weight
            )
        
        self.criterion.to(self.device)
    
    def _setup_dataloader(self):
        """Setup data loader"""
        self.logger.info(f"Loading dataset from: {self.args.data_path}")
        
        dataset = MultiExposureDynamicLoader(
            dataset_path=self.args.data_path,
            mode='train',
            image_size=(1200, 900),
            min_exposures=1,
            max_exposures=None,
            fixed_n=None
        )
        
        self.train_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.num_workers,
            pin_memory=True if self.device.type == 'cuda' else False,
            collate_fn=collate_multi_exposure
        )
        
        self.logger.info(f"Loaded {len(dataset)} image sets")
    
    def _save_checkpoint(self, epoch, is_best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'encoder_state_dict': self.encoder.state_dict(),
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'args': self.args
        }
        
        checkpoint_dir = self.args.save_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Regular checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f'epoch_{epoch}_checkpoint.pth')
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Best checkpoint
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")
    
    def _load_checkpoint(self, checkpoint_path):
        """Load checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.logger.info(f"✓ Checkpoint loaded (encoder + model): {checkpoint_path}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.encoder.train()
        self.model.train()
        total_loss = 0
        loss_dict_accumulate = {}
        
        pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader), 
                   desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        
        for batch_idx, batch in pbar:
            if isinstance(batch, dict):
                exposures_list = batch.get('multi_exposure', None)
                img_gt = batch.get('label', batch.get('gt', None))
            else:
                raise ValueError("Batch must be a dictionary")
            
            exposures_list = [exp.to(self.device) for exp in exposures_list]
            if img_gt is not None:
                img_gt = img_gt.to(self.device)
            
            self.optimizer.zero_grad()
            
            fused_features = self.encoder(exposures_list)
            mul, add, img_high = self.model(fused_features)
            
            if self.args.loss_type == 'Combined' and img_gt is not None:
                loss, loss_dict = self.criterion(img_high, img_gt)
            else:
                loss, loss_dict = self.criterion(img_high, img_gt if img_gt is not None else None)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.model.parameters()), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_dict_accumulate:
                    loss_dict_accumulate[key] = 0
                loss_dict_accumulate[key] += value.item()
            
            self.global_step += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        epoch_loss = total_loss / (batch_idx + 1)
        for key in loss_dict_accumulate:
            loss_dict_accumulate[key] /= (batch_idx + 1)
        
        return epoch_loss, loss_dict_accumulate
    
    def train(self):
        """Main training loop"""
        self.logger.info("Starting training...")
        self.logger.info(f"Config: {vars(self.args)}")
        
        for epoch in range(self.args.num_epochs):
            epoch_loss, loss_dict = self.train_epoch(epoch)
            
            self.logger.info(f"Epoch {epoch+1}/{self.args.num_epochs} - Loss: {epoch_loss:.4f}")
            for key, value in loss_dict.items():
                self.logger.info(f"  {key}: {value:.4f}")
            
            self.scheduler.step()
            
            current_lr = self.optimizer.param_groups[0]['lr']
            self.logger.info(f"  LR: {current_lr:.6f}")
            
            if (epoch + 1) % self.args.save_interval == 0:
                is_best = epoch_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = epoch_loss
                self._save_checkpoint(epoch + 1, is_best=is_best)
        
        self.logger.info("Training completed!")


def main():
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    # Create trainer and train
    trainer = EnhancedIATTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
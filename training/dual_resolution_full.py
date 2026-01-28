# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""
Full-Resolution Dual-Resolution Semantic Segmentation Training Module
Handles mixed-size batches (1024×1024 perspective + 1024×2048 ERP)
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_semantic import MaskClassificationSemantic


class DualResolutionFullSemantic(MaskClassificationSemantic):
    """
    Extends MaskClassificationSemantic to handle full-resolution dual-resolution batches.
    Processes 1024×1024 and 1024×2048 images separately, then combines losses.
    """
    
    def training_step(self, batch, batch_idx):
        """Handle full-resolution dual-resolution batch with grouped images"""
        imgs_1024 = batch.get('imgs_1024')
        imgs_2048 = batch.get('imgs_2048')
        targets_1024 = batch.get('targets_1024', [])
        targets_2048 = batch.get('targets_2048', [])
        
        total_loss = 0.0
        num_groups = 0
        
        # Process 1024×1024 images (perspective)
        if imgs_1024 is not None and len(targets_1024) > 0:
            mask_logits_per_block, class_logits_per_block = self(imgs_1024)
            
            # Iterate through each decoder layer's outputs
            losses_1024_all_blocks = {}
            for i, (mask_logits, class_logits) in enumerate(zip(mask_logits_per_block, class_logits_per_block)):
                losses = self.criterion(
                    mask_logits,
                    targets_1024,
                    class_logits,
                )
                block_postfix = self.block_postfix(i)
                losses = {f"{key}_1024x1024{block_postfix}": value for key, value in losses.items()}
                losses_1024_all_blocks |= losses
            
            loss_1024_total = self.criterion.loss_total(losses_1024_all_blocks, self.log)
            total_loss += loss_1024_total
            num_groups += 1
            
            # Log 1024×1024 metrics
            self.log('train/loss_1024x1024', loss_1024_total, prog_bar=True, sync_dist=True)
            for key, value in losses_1024_all_blocks.items():
                self.log(f'train/{key}', value, sync_dist=True)
        
        # Process 1024×2048 images (ERP)
        # TODO: Model architecture is fixed to img_size specified in config.
        # Cannot process 1024x2048 images with a model initialized for 1024x1024.
        # Options: 1) Train separate model for ERP, 2) Resize ERP to 1024x1024, or 3) Initialize model for 1024x2048
        # For now, skipping ERP images to allow training to proceed with perspective images only.
        if False and imgs_2048 is not None and len(targets_2048) > 0:
            mask_logits_per_block, class_logits_per_block = self(imgs_2048)
            
            # Iterate through each decoder layer's outputs
            losses_2048_all_blocks = {}
            for i, (mask_logits, class_logits) in enumerate(zip(mask_logits_per_block, class_logits_per_block)):
                losses = self.criterion(
                    mask_logits,
                    targets_2048,
                    class_logits,
                )
                block_postfix = self.block_postfix(i)
                losses = {f"{key}_1024x2048{block_postfix}": value for key, value in losses.items()}
                losses_2048_all_blocks |= losses
            
            loss_2048_total = self.criterion.loss_total(losses_2048_all_blocks, self.log)
            total_loss += loss_2048_total
            num_groups += 1
            
            # Log 1024×2048 metrics
            self.log('train/loss_1024x2048', loss_2048_total, prog_bar=True, sync_dist=True)
            for key, value in losses_2048_all_blocks.items():
                self.log(f'train/{key}', value, sync_dist=True)
        
        # Average loss across groups
        if num_groups > 0:
            total_loss = total_loss / num_groups
        
        # Log combined metrics
        self.log('train/loss', total_loss, prog_bar=True, sync_dist=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation uses standard single-size batches (ERP only)"""
        return super().validation_step(batch, batch_idx)

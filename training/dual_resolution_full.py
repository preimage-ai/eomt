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


class DualResolutionFull(MaskClassificationSemantic):
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
            mask_logits_1024, class_logits_1024 = self(imgs_1024)
            loss_1024 = self.criterion(
                mask_logits_1024,
                class_logits_1024,
                targets_1024,
            )
            total_loss += loss_1024['loss']
            num_groups += 1
            
            # Log 1024×1024 metrics
            self.log('train/loss_1024x1024', loss_1024['loss'], prog_bar=True, sync_dist=True)
            self.log('train/mask_loss_1024x1024', loss_1024['mask_loss'], sync_dist=True)
            self.log('train/dice_loss_1024x1024', loss_1024['dice_loss'], sync_dist=True)
            self.log('train/class_loss_1024x1024', loss_1024['class_loss'], sync_dist=True)
        
        # Process 1024×2048 images (ERP)
        if imgs_2048 is not None and len(targets_2048) > 0:
            mask_logits_2048, class_logits_2048 = self(imgs_2048)
            loss_2048 = self.criterion(
                mask_logits_2048,
                class_logits_2048,
                targets_2048,
            )
            total_loss += loss_2048['loss']
            num_groups += 1
            
            # Log 1024×2048 metrics
            self.log('train/loss_1024x2048', loss_2048['loss'], prog_bar=True, sync_dist=True)
            self.log('train/mask_loss_1024x2048', loss_2048['mask_loss'], sync_dist=True)
            self.log('train/dice_loss_1024x2048', loss_2048['dice_loss'], sync_dist=True)
            self.log('train/class_loss_1024x2048', loss_2048['class_loss'], sync_dist=True)
        
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

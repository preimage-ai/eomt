# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""
Dual-Resolution Semantic Segmentation Training Module
Handles mixed-size batches (512×512 perspective + 512×1024 ERP)
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.mask_classification_semantic import MaskClassificationSemantic


class DualResolutionSemantic(MaskClassificationSemantic):
    """
    Extends MaskClassificationSemantic to handle dual-resolution batches.
    Processes 512×512 and 512×1024 images separately, then combines losses.
    """
    
    def training_step(self, batch, batch_idx):
        """Handle dual-resolution batch with grouped images"""
        imgs_512 = batch.get('imgs_512')
        imgs_1024 = batch.get('imgs_1024')
        targets_512 = batch.get('targets_512', [])
        targets_1024 = batch.get('targets_1024', [])
        
        total_loss = 0.0
        num_groups = 0
        
        # Process 512×512 images
        if imgs_512 is not None and len(targets_512) > 0:
            mask_logits_512, class_logits_512 = self(imgs_512)
            loss_512 = self.criterion(
                mask_logits_512,
                class_logits_512,
                targets_512,
            )
            total_loss += loss_512['loss']
            num_groups += 1
            
            # Log 512×512 metrics
            self.log('train/loss_512', loss_512['loss'], prog_bar=True, sync_dist=True)
            self.log('train/mask_loss_512', loss_512['mask_loss'], sync_dist=True)
            self.log('train/dice_loss_512', loss_512['dice_loss'], sync_dist=True)
            self.log('train/class_loss_512', loss_512['class_loss'], sync_dist=True)
        
        # Process 512×1024 images
        if imgs_1024 is not None and len(targets_1024) > 0:
            mask_logits_1024, class_logits_1024 = self(imgs_1024)
            loss_1024 = self.criterion(
                mask_logits_1024,
                class_logits_1024,
                targets_1024,
            )
            total_loss += loss_1024['loss']
            num_groups += 1
            
            # Log 512×1024 metrics
            self.log('train/loss_1024', loss_1024['loss'], prog_bar=True, sync_dist=True)
            self.log('train/mask_loss_1024', loss_1024['mask_loss'], sync_dist=True)
            self.log('train/dice_loss_1024', loss_1024['dice_loss'], sync_dist=True)
            self.log('train/class_loss_1024', loss_1024['class_loss'], sync_dist=True)
        
        # Average loss across groups
        if num_groups > 0:
            total_loss = total_loss / num_groups
        
        # Log combined metrics
        self.log('train/loss', total_loss, prog_bar=True, sync_dist=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], prog_bar=True, sync_dist=True)
        
        return total_loss
    
    def validation_step(self, batch, batch_idx):
        """Validation uses standard single-size batches (ERP only)"""
        # Validation dataset returns standard batches, not grouped
        return super().validation_step(batch, batch_idx)

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
            mask_logits_per_block, class_logits_per_block = self(imgs_512)
            
            # Iterate through each decoder layer's outputs
            losses_512_all_blocks = {}
            for i, (mask_logits, class_logits) in enumerate(zip(mask_logits_per_block, class_logits_per_block)):
                losses = self.criterion(
                    mask_logits,
                    targets_512,
                    class_logits,
                )
                block_postfix = self.block_postfix(i)
                losses = {f"{key}_512{block_postfix}": value for key, value in losses.items()}
                losses_512_all_blocks |= losses
            
            loss_512_total = self.criterion.loss_total(losses_512_all_blocks, self.log)
            total_loss += loss_512_total
            num_groups += 1
            
            # Log 512×512 metrics
            self.log('train/loss_512', loss_512_total, prog_bar=True, sync_dist=True)
            for key, value in losses_512_all_blocks.items():
                self.log(f'train/{key}', value, sync_dist=True)
        
        # Process 512×1024 images
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
                losses = {f"{key}_1024{block_postfix}": value for key, value in losses.items()}
                losses_1024_all_blocks |= losses
            
            loss_1024_total = self.criterion.loss_total(losses_1024_all_blocks, self.log)
            total_loss += loss_1024_total
            num_groups += 1
            
            # Log 512×1024 metrics
            self.log('train/loss_1024', loss_1024_total, prog_bar=True, sync_dist=True)
            for key, value in losses_1024_all_blocks.items():
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
        # Validation dataset returns standard batches, not grouped
        return super().validation_step(batch, batch_idx)

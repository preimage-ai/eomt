# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union, Tuple, List
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms

CLASS_MAPPING = {i: i - 1 for i in range(1, 167)}


class GradualResolutionERP(LightningDataModule):
    """
    Dataset for gradual resolution adaptation from cubemap to full ERP.
    
    Phases:
    1. (512, 512) - Cubemap-like crops from ERP
    2. (512, 768) - Intermediate resolution
    3. (512, 1024) - Full ERP resolution
    
    Automatically switches resolution based on current epoch.
    """
    def __init__(
        self,
        erp_image_dir: str,
        erp_mask_dir: str,
        num_workers: int = 2,
        batch_size: int = 4,
        final_img_size: Tuple[int, int] = (512, 1024),
        num_classes: int = 166,
        color_jitter_enabled: bool = True,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        check_empty_targets: bool = True,
        erp_augmentation_factor: int = 20,
        phase_epochs: List[int] = [10, 20, 20],  # Epochs per phase
    ) -> None:
        super().__init__(
            path=erp_image_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=final_img_size,
            check_empty_targets=check_empty_targets,
        )
        
        self.erp_image_dir = Path(erp_image_dir)
        self.erp_mask_dir = Path(erp_mask_dir)
        self.final_img_size = final_img_size
        self.erp_augmentation_factor = erp_augmentation_factor
        self.phase_epochs = phase_epochs
        
        # Define resolution phases
        self.resolutions = [
            (512, 512),   # Phase 1: Cubemap-like
            (512, 768),   # Phase 2: Intermediate
            (512, 1024),  # Phase 3: Full ERP
        ]
        
        self.current_phase = 0
        self.current_resolution = self.resolutions[0]
        
        print(f"Gradual Resolution ERP Dataset:")
        print(f"  - Phase 1 ({phase_epochs[0]} epochs): {self.resolutions[0]}")
        print(f"  - Phase 2 ({phase_epochs[1]} epochs): {self.resolutions[1]}")
        print(f"  - Phase 3 ({phase_epochs[2]} epochs): {self.resolutions[2]}")
        print(f"  - Augmentation factor: {erp_augmentation_factor}x")
        
        self.save_hyperparameters(ignore=["_class_path"])

        # Create transforms for each phase
        self.transforms_by_phase = [
            ERPAugmentedTransforms(
                img_size=res,
                color_jitter_enabled=color_jitter_enabled,
                scale_range=scale_range,
                horizontal_wrap=True,
                vertical_shift=True,
            )
            for res in self.resolutions
        ]

    @staticmethod
    def target_parser(target, **kwargs):
        masks, labels = [], []

        for label_id in target[0].unique():
            cls_id = label_id.item()

            if cls_id not in CLASS_MAPPING:
                continue

            masks.append(target[0] == label_id)
            labels.append(CLASS_MAPPING[cls_id])

        return masks, labels, [False for _ in range(len(masks))]

    def update_phase(self, current_epoch: int):
        """Update resolution phase based on current epoch"""
        cumulative_epochs = 0
        for phase_idx, phase_length in enumerate(self.phase_epochs):
            cumulative_epochs += phase_length
            if current_epoch < cumulative_epochs:
                if phase_idx != self.current_phase:
                    self.current_phase = phase_idx
                    self.current_resolution = self.resolutions[phase_idx]
                    print(f"\n{'='*60}")
                    print(f"Switching to Phase {phase_idx + 1}: {self.current_resolution}")
                    print(f"{'='*60}\n")
                    # Recreate datasets with new resolution
                    self.setup()
                return
        
        # If past all phases, stay at final resolution
        if self.current_phase != len(self.resolutions) - 1:
            self.current_phase = len(self.resolutions) - 1
            self.current_resolution = self.resolutions[-1]
            print(f"\n{'='*60}")
            print(f"Final Phase: {self.current_resolution}")
            print(f"{'='*60}\n")
            self.setup()

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        # Load ERP images
        erp_images = sorted(list(self.erp_image_dir.glob("*.jpg")) + 
                           list(self.erp_image_dir.glob("*.png")))
        erp_masks = [self.erp_mask_dir / f"{img.stem}.png" for img in erp_images]
        
        # Filter valid pairs
        valid_pairs = [(img, mask) for img, mask in zip(erp_images, erp_masks) if mask.exists()]
        
        if len(valid_pairs) == 0:
            raise ValueError(f"No ERP images found in {self.erp_image_dir}")
        
        # Split train/val
        val_split = max(1, len(valid_pairs) // 10)
        train_pairs = valid_pairs[val_split:]
        val_pairs = valid_pairs[:val_split]
        
        # Create datasets with current phase resolution
        self.train_dataset = ERPAugmentedDataset(
            train_pairs,
            transforms=self.transforms_by_phase[self.current_phase],
            augmentation_factor=self.erp_augmentation_factor,
            target_parser=self.target_parser,
        )
        
        self.val_dataset = ERPAugmentedDataset(
            val_pairs,
            transforms=None,
            augmentation_factor=1,
            target_parser=self.target_parser,
        )

        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )


class ERPAugmentedDataset(torch.utils.data.Dataset):
    """Dataset that applies heavy augmentation to limited ERP samples"""
    
    def __init__(self, image_mask_pairs, transforms, augmentation_factor, target_parser):
        self.pairs = image_mask_pairs
        self.transforms = transforms
        self.augmentation_factor = augmentation_factor
        self.target_parser = target_parser
        
    def __len__(self):
        return len(self.pairs) * self.augmentation_factor
    
    def __getitem__(self, idx):
        # Map augmented index back to original image
        original_idx = idx % len(self.pairs)
        img_path, mask_path = self.pairs[original_idx]
        
        # Load image and mask
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)
        
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0)
        
        # Apply transforms
        if self.transforms:
            img, mask = self.transforms(img, mask)
        
        # Parse target
        masks, labels, is_crowd = self.target_parser(mask)
        
        return img, (masks, labels, is_crowd)


class ERPAugmentedTransforms:
    """Augmentation optimized for cubemap-to-ERP adaptation"""
    
    def __init__(
        self,
        img_size,
        color_jitter_enabled=True,
        scale_range=(0.8, 1.2),
        horizontal_wrap=True,
        vertical_shift=True,
    ):
        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.scale_range = scale_range
        self.horizontal_wrap = horizontal_wrap
        self.vertical_shift = vertical_shift
        
        # Use standard transforms as base
        self.base_transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
    
    def __call__(self, img, mask):
        # Apply base transforms first
        img, mask = self.base_transforms(img, mask)
        
        # Panorama-specific augmentations
        if self.horizontal_wrap and random.random() < 0.8:
            # Horizontal wrapping (critical for panoramas)
            shift = random.randint(0, img.shape[-1])
            img = torch.roll(img, shift, dims=-1)
            mask = torch.roll(mask, shift, dims=-1)
        
        if self.vertical_shift and random.random() < 0.5:
            # Vertical shift (simulate different camera heights)
            shift = random.randint(-img.shape[-2] // 6, img.shape[-2] // 6)
            img = torch.roll(img, shift, dims=-2)
            mask = torch.roll(mask, shift, dims=-2)
        
        # Light color augmentation (already learned in cubemap phase)
        if self.color_jitter_enabled and random.random() < 0.3:
            factor = random.uniform(0.9, 1.1)
            img = torch.clamp(img * factor, 0, 1)
        
        return img, mask

# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union, Tuple, Optional
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from PIL import Image

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms

CLASS_MAPPING = {i: i - 1 for i in range(1, 167)}


class DualResolutionMixed(LightningDataModule):
    """
    Dual-resolution training with mixed batches.
    
    Strategy:
    - 70% perspective images (512×512) from ADE20K
    - 30% ERP images (512×1024) with heavy augmentation
    - Mixed within each batch for stable gradients
    - Multi-scale ViT handles both resolutions
    
    Key improvements:
    1. Single dataloader with mixed sampling (not two separate loaders)
    2. Batch-level mixing ensures every batch has both resolutions
    3. Rooftop-only classes (150-165) masked for perspective samples
    4. Heavy ERP augmentation (30x) to balance with perspective data
    """
    def __init__(
        self,
        perspective_path: str,  # ADE20K path
        erp_image_dir: str,
        erp_mask_dir: str,
        num_workers: int = 4,
        batch_size: int = 8,  # Larger batch for mixed sampling
        perspective_img_size: Tuple[int, int] = (512, 512),
        erp_img_size: Tuple[int, int] = (512, 1024),
        num_classes: int = 166,
        color_jitter_enabled: bool = True,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
        perspective_ratio: float = 0.7,  # 70% perspective, 30% ERP per batch
        erp_augmentation_factor: int = 30,  # Repeat ERP 30x
    ) -> None:
        super().__init__(
            path=perspective_path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=erp_img_size,  # Use larger size as base
            check_empty_targets=check_empty_targets,
        )
        
        self.perspective_path = Path(perspective_path)
        self.erp_image_dir = Path(erp_image_dir)
        self.erp_mask_dir = Path(erp_mask_dir)
        self.perspective_img_size = perspective_img_size
        self.erp_img_size = erp_img_size
        self.perspective_ratio = perspective_ratio
        self.erp_augmentation_factor = erp_augmentation_factor
        
        # Calculate samples per batch
        self.perspective_per_batch = int(batch_size * perspective_ratio)
        self.erp_per_batch = batch_size - self.perspective_per_batch
        
        print(f"\n{'='*60}")
        print(f"Dual-Resolution Mixed Training")
        print(f"{'='*60}")
        print(f"Perspective:")
        print(f"  - Path: {perspective_path}")
        print(f"  - Resolution: {perspective_img_size}")
        print(f"  - Samples per batch: {self.perspective_per_batch}")
        print(f"ERP:")
        print(f"  - Path: {erp_image_dir}")
        print(f"  - Resolution: {erp_img_size}")
        print(f"  - Augmentation: {erp_augmentation_factor}x")
        print(f"  - Samples per batch: {self.erp_per_batch}")
        print(f"Batch composition: {self.perspective_per_batch} perspective + {self.erp_per_batch} ERP = {batch_size} total")
        print(f"{'='*60}\n")
        
        self.save_hyperparameters(ignore=["_class_path"])

        # Create transforms for each resolution
        self.perspective_transforms = Transforms(
            img_size=perspective_img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        
        self.erp_transforms = ERPAugmentedTransforms(
            img_size=erp_img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
            horizontal_wrap=True,
            vertical_shift=True,
        )

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

    def setup(self, stage: Union[str, None] = None) -> LightningDataModule:
        # Load perspective dataset (ADE20K)
        perspective_dataset = Dataset(
            path=str(self.perspective_path),
            split="training",
            transforms=self.perspective_transforms,
            target_parser=self.target_parser,
        )
        
        # Load ERP images
        erp_images = sorted(list(self.erp_image_dir.glob("*.jpg")) + 
                           list(self.erp_image_dir.glob("*.png")))
        erp_masks = [self.erp_mask_dir / f"{img.stem}.png" for img in erp_images]
        
        # Filter valid pairs
        erp_pairs = [(img, mask) for img, mask in zip(erp_images, erp_masks) if mask.exists()]
        
        if len(erp_pairs) == 0:
            raise ValueError(f"No ERP images found in {self.erp_image_dir}")
        
        print(f"Found {len(perspective_dataset)} perspective images")
        print(f"Found {len(erp_pairs)} ERP images (will be augmented {self.erp_augmentation_factor}x)")
        
        # Split ERP into train/val
        val_split = max(1, len(erp_pairs) // 10)
        erp_train_pairs = erp_pairs[val_split:]
        erp_val_pairs = erp_pairs[:val_split]
        
        # Create ERP datasets with augmentation
        erp_train_dataset = ERPAugmentedDataset(
            erp_train_pairs,
            transforms=self.erp_transforms,
            augmentation_factor=self.erp_augmentation_factor,
            target_parser=self.target_parser,
        )
        
        erp_val_dataset = ERPAugmentedDataset(
            erp_val_pairs,
            transforms=None,  # No augmentation for validation
            augmentation_factor=1,
            target_parser=self.target_parser,
        )
        
        # Create mixed datasets
        self.train_dataset = MixedResolutionDataset(
            perspective_dataset=perspective_dataset,
            erp_dataset=erp_train_dataset,
            perspective_ratio=self.perspective_ratio,
            perspective_img_size=self.perspective_img_size,
            erp_img_size=self.erp_img_size,
        )
        
        # For validation, use only ERP (we care about ERP performance)
        self.val_dataset = erp_val_dataset

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


class MixedResolutionDataset(TorchDataset):
    """
    Dataset that mixes perspective and ERP samples with different resolutions.
    
    Each batch will contain:
    - perspective_ratio * batch_size perspective images (512×512)
    - (1 - perspective_ratio) * batch_size ERP images (512×1024)
    
    Returns metadata to indicate source for class weight masking.
    """
    
    def __init__(
        self,
        perspective_dataset,
        erp_dataset,
        perspective_ratio: float,
        perspective_img_size: Tuple[int, int],
        erp_img_size: Tuple[int, int],
    ):
        self.perspective_dataset = perspective_dataset
        self.erp_dataset = erp_dataset
        self.perspective_ratio = perspective_ratio
        self.perspective_img_size = perspective_img_size
        self.erp_img_size = erp_img_size
        
        # Use the larger dataset as base length
        self.length = max(len(perspective_dataset), len(erp_dataset))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Randomly choose perspective or ERP based on ratio
        if random.random() < self.perspective_ratio:
            # Get perspective sample
            img, target = self.perspective_dataset[idx % len(self.perspective_dataset)]
            source = "perspective"
            img_size = self.perspective_img_size
        else:
            # Get ERP sample
            img, target = self.erp_dataset[idx % len(self.erp_dataset)]
            source = "erp"
            img_size = self.erp_img_size
        
        # Add metadata for class weight masking
        # The loss function will use this to mask rooftop-only classes for perspective samples
        masks, labels, is_crowd = target
        
        # Add source metadata (will be used by loss function)
        metadata = {
            'source': source,
            'img_size': img_size,
        }
        
        return img, (masks, labels, is_crowd, metadata)


class ERPAugmentedDataset(TorchDataset):
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
    """Augmentation optimized for ERP panoramas"""
    
    def __init__(
        self,
        img_size,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
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
        
        return img, mask

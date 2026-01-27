# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union, Tuple
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from PIL import Image

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms

CLASS_MAPPING = {i: i - 1 for i in range(1, 167)}


class DualResolutionTrain(LightningDataModule):
    """
    Dual-resolution training from scratch with mixed batches.
    
    Strategy:
    - Perspective images (512×512) from ADE20K zip
    - ERP images (1024×2048) from separate folder
    - Mixed within each batch for multi-scale learning
    - Multi-scale ViT handles both resolutions
    
    Key features:
    1. Single dataloader with mixed sampling
    2. Batch-level mixing for stable gradients
    3. Rooftop-only classes (150-165) masked for perspective samples
    4. Supports zipped ADE20K dataset
    """
    def __init__(
        self,
        perspective_path: str,  # Path to ADE20K zip folder
        erp_image_dir: str,
        erp_mask_dir: str,
        num_workers: int = 4,
        batch_size: int = 8,
        perspective_img_size: Tuple[int, int] = (512, 512),
        erp_img_size: Tuple[int, int] = (1024, 2048),
        num_classes: int = 166,
        color_jitter_enabled: bool = True,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
        perspective_ratio: float = 0.7,  # 70% perspective, 30% ERP per batch
        erp_augmentation_factor: int = 50,  # Repeat ERP samples for balance
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
        
        print(f"\n{'='*70}")
        print(f"Dual-Resolution Training from Scratch")
        print(f"{'='*70}")
        print(f"Perspective (ADE20K):")
        print(f"  - Path: {perspective_path}")
        print(f"  - Resolution: {perspective_img_size}")
        print(f"  - Samples per batch: {self.perspective_per_batch}")
        print(f"ERP:")
        print(f"  - Path: {erp_image_dir}")
        print(f"  - Resolution: {erp_img_size}")
        print(f"  - Augmentation: {erp_augmentation_factor}x")
        print(f"  - Samples per batch: {self.erp_per_batch}")
        print(f"Batch composition: {self.perspective_per_batch} perspective + {self.erp_per_batch} ERP = {batch_size} total")
        print(f"{'='*70}\n")
        
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
        # Load perspective dataset from ADE20K zip
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "zip_path": Path(self.perspective_path, "ADEChallengeData2016.zip"),
            "target_zip_path": Path(self.perspective_path, "ADEChallengeData2016.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        
        perspective_train_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/training"),
            target_folder_path_in_zip=Path("./ADEChallengeData2016/annotations/training"),
            transforms=self.perspective_transforms,
            **dataset_kwargs,
        )
        
        perspective_val_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/validation"),
            target_folder_path_in_zip=Path("./ADEChallengeData2016/annotations/validation"),
            transforms=None,  # No augmentation for validation
            **dataset_kwargs,
        )
        
        # Load ERP images from folder
        erp_images = sorted(list(self.erp_image_dir.glob("*.jpg")) + 
                           list(self.erp_image_dir.glob("*.png")))
        erp_masks = [self.erp_mask_dir / f"{img.stem}.png" for img in erp_images]
        
        # Filter valid pairs
        erp_pairs = [(img, mask) for img, mask in zip(erp_images, erp_masks) if mask.exists()]
        
        if len(erp_pairs) == 0:
            raise ValueError(f"No ERP images found in {self.erp_image_dir}")
        
        print(f"Found {len(perspective_train_dataset)} perspective training images")
        print(f"Found {len(perspective_val_dataset)} perspective validation images")
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
            perspective_dataset=perspective_train_dataset,
            erp_dataset=erp_train_dataset,
            perspective_ratio=self.perspective_ratio,
            perspective_img_size=self.perspective_img_size,
            erp_img_size=self.erp_img_size,
        )
        
        # For validation, create mixed dataset too
        self.val_dataset = MixedResolutionDataset(
            perspective_dataset=perspective_val_dataset,
            erp_dataset=erp_val_dataset,
            perspective_ratio=self.perspective_ratio,
            perspective_img_size=self.perspective_img_size,
            erp_img_size=self.erp_img_size,
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


class MixedResolutionDataset(TorchDataset):
    """
    Dataset that mixes perspective and ERP samples with different resolutions.
    
    Each batch will contain:
    - perspective_ratio * batch_size perspective images (512×512)
    - (1 - perspective_ratio) * batch_size ERP images (1024×2048)
    
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
        masks, labels, is_crowd = target
        
        # Add source metadata (will be used by loss function)
        metadata = {
            'source': source,
            'img_size': img_size,
        }
        
        return img, (masks, labels, is_crowd, metadata)


class ERPAugmentedDataset(TorchDataset):
    """Dataset that applies heavy augmentation to ERP samples"""
    
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

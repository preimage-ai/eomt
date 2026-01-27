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


class DualResolutionWindowed(LightningDataModule):
    """
    Dual-resolution training with windowed ERP crops.
    
    Strategy:
    - Perspective: Standard 512×512 images from ADE20K
    - ERP: Extract random 512×512 crops from native resolution (e.g., 1024×2048)
    - Mixed within each batch
    - Preserves ERP native resolution and aspect ratio
    
    Key improvements:
    1. No squishing of ERP images
    2. Trains on native ERP resolution via windowing
    3. Each epoch sees different crops from ERP images
    4. Panorama-specific augmentations still applied
    """
    def __init__(
        self,
        perspective_path: str,  # ADE20K path
        erp_image_dir: str,
        erp_mask_dir: str,
        num_workers: int = 4,
        batch_size: int = 16,
        perspective_img_size: Tuple[int, int] = (512, 512),
        erp_crop_size: Tuple[int, int] = (512, 512),  # Crop size from ERP
        num_classes: int = 166,
        color_jitter_enabled: bool = True,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
        perspective_ratio: float = 0.95,  # 95% perspective, 5% ERP
        erp_augmentation_factor: int = 200,  # Each ERP image produces many crops
    ) -> None:
        super().__init__(
            path=perspective_path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=perspective_img_size,
            check_empty_targets=check_empty_targets,
        )
        
        self.perspective_path = Path(perspective_path)
        self.erp_image_dir = Path(erp_image_dir)
        self.erp_mask_dir = Path(erp_mask_dir)
        self.perspective_img_size = perspective_img_size
        self.erp_crop_size = erp_crop_size
        self.perspective_ratio = perspective_ratio
        self.erp_augmentation_factor = erp_augmentation_factor
        
        # Calculate samples per batch
        self.perspective_per_batch = int(batch_size * perspective_ratio)
        self.erp_per_batch = batch_size - self.perspective_per_batch
        
        print(f"\n{'='*70}")
        print(f"Dual-Resolution Windowed Training")
        print(f"{'='*70}")
        print(f"Perspective:")
        print(f"  - Path: {perspective_path}")
        print(f"  - Resolution: {perspective_img_size}")
        print(f"  - Samples per batch: {self.perspective_per_batch}")
        print(f"ERP (Windowed):")
        print(f"  - Path: {erp_image_dir}")
        print(f"  - Crop size: {erp_crop_size} (extracted from native resolution)")
        print(f"  - Augmentation: {erp_augmentation_factor}x crops per image")
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
        
        # ERP transforms without resizing (crops are already correct size)
        self.erp_transforms = ERPWindowedTransforms(
            crop_size=erp_crop_size,
            color_jitter_enabled=color_jitter_enabled,
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
        
        if len(erp_images) == 0:
            raise ValueError(
                f"No ERP images found in {self.erp_image_dir}\n"
                f"Please ensure the directory exists and contains .jpg or .png files"
            )
        
        erp_masks = [self.erp_mask_dir / f"{img.stem}_mask_ids.png" for img in erp_images]
        
        # Filter valid pairs
        erp_pairs = [(img, mask) for img, mask in zip(erp_images, erp_masks) if mask.exists()]
        
        if len(erp_pairs) == 0:
            missing_masks = [mask for mask in erp_masks if not mask.exists()]
            raise ValueError(
                f"Found {len(erp_images)} ERP images in {self.erp_image_dir}, "
                f"but no corresponding masks in {self.erp_mask_dir}\n"
                f"Missing masks for: {[m.name for m in missing_masks[:5]]}...\n"
                f"Expected mask format: <image_stem>_mask_ids.png"
            )
        
        print(f"Found {len(perspective_train_dataset)} perspective training images")
        print(f"Found {len(perspective_val_dataset)} perspective validation images")
        print(f"Found {len(erp_pairs)} ERP images (will extract {self.erp_augmentation_factor} crops each)")
        
        # Split ERP into train/val
        val_split = max(1, len(erp_pairs) // 10)
        erp_train_pairs = erp_pairs[val_split:]
        erp_val_pairs = erp_pairs[:val_split]
        
        # Create ERP datasets with windowed cropping
        erp_train_dataset = ERPWindowedDataset(
            erp_train_pairs,
            crop_size=self.erp_crop_size,
            transforms=self.erp_transforms,
            augmentation_factor=self.erp_augmentation_factor,
            target_parser=self.target_parser,
        )
        
        erp_val_dataset = ERPWindowedDataset(
            erp_val_pairs,
            crop_size=self.erp_crop_size,
            transforms=None,  # No augmentation for validation
            augmentation_factor=1,
            target_parser=self.target_parser,
        )
        
        # Create mixed datasets
        self.train_dataset = MixedResolutionDataset(
            perspective_dataset=perspective_train_dataset,
            erp_dataset=erp_train_dataset,
            perspective_ratio=self.perspective_ratio,
        )
        
        # For validation, use mixed dataset too
        self.val_dataset = MixedResolutionDataset(
            perspective_dataset=perspective_val_dataset,
            erp_dataset=erp_val_dataset,
            perspective_ratio=self.perspective_ratio,
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
    """Dataset that mixes perspective and ERP samples"""
    
    def __init__(
        self,
        perspective_dataset,
        erp_dataset,
        perspective_ratio: float,
    ):
        self.perspective_dataset = perspective_dataset
        self.erp_dataset = erp_dataset
        self.perspective_ratio = perspective_ratio
        
        # Use the larger dataset as base length
        self.length = max(len(perspective_dataset), len(erp_dataset))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Randomly choose perspective or ERP based on ratio
        if random.random() < self.perspective_ratio:
            # Get perspective sample
            img, target = self.perspective_dataset[idx % len(self.perspective_dataset)]
        else:
            # Get ERP sample (already cropped to 512x512)
            img, target = self.erp_dataset[idx % len(self.erp_dataset)]
        
        return img, target


class ERPWindowedDataset(TorchDataset):
    """
    Dataset that extracts random 512×512 crops from native resolution ERP images.
    
    Each ERP image produces multiple crops via augmentation_factor.
    Crops are extracted randomly, preserving native resolution.
    """
    
    def __init__(self, image_mask_pairs, crop_size, transforms, augmentation_factor, target_parser):
        self.pairs = image_mask_pairs
        self.crop_size = crop_size
        self.transforms = transforms
        self.augmentation_factor = augmentation_factor
        self.target_parser = target_parser
        
    def __len__(self):
        return len(self.pairs) * self.augmentation_factor
    
    def __getitem__(self, idx):
        # Map augmented index back to original image
        original_idx = idx % len(self.pairs)
        img_path, mask_path = self.pairs[original_idx]
        
        # Load image and mask at native resolution
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Load as grayscale
        
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).long()  # Single channel, long dtype
        
        # Extract random crop
        img, mask = self._random_crop(img, mask, self.crop_size)
        
        # Apply transforms (augmentations, no resizing)
        if self.transforms:
            img, mask = self.transforms(img, mask)
        
        # Parse target
        masks, labels, is_crowd = self.target_parser(mask)
        
        # Convert to dict format expected by validation
        target = {
            "masks": torch.stack(masks) if masks else torch.zeros((0, *mask.shape[-2:])),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }
        
        return img, target
    
    def _random_crop(self, img, mask, crop_size):
        """Extract random crop from image and mask"""
        _, h, w = img.shape
        crop_h, crop_w = crop_size
        
        # If image is smaller than crop size, pad it
        if h < crop_h or w < crop_w:
            pad_h = max(0, crop_h - h)
            pad_w = max(0, crop_w - w)
            img = torch.nn.functional.pad(img, (0, pad_w, 0, pad_h), mode='reflect')
            mask = torch.nn.functional.pad(mask, (0, pad_w, 0, pad_h), mode='reflect')
            _, h, w = img.shape
        
        # Random crop position
        top = random.randint(0, h - crop_h)
        left = random.randint(0, w - crop_w)
        
        img_crop = img[:, top:top+crop_h, left:left+crop_w]
        mask_crop = mask[:, top:top+crop_h, left:left+crop_w]
        
        return img_crop, mask_crop


class ERPWindowedTransforms:
    """Augmentation for windowed ERP crops (no resizing needed)"""
    
    def __init__(
        self,
        crop_size,
        color_jitter_enabled=True,
        horizontal_wrap=True,
        vertical_shift=True,
    ):
        self.crop_size = crop_size
        self.color_jitter_enabled = color_jitter_enabled
        self.horizontal_wrap = horizontal_wrap
        self.vertical_shift = vertical_shift
    
    def __call__(self, img, mask):
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
        
        # Light color augmentation
        if self.color_jitter_enabled and random.random() < 0.5:
            factor = random.uniform(0.9, 1.1)
            img = torch.clamp(img * factor, 0, 1)
        
        # Random horizontal flip
        if random.random() < 0.5:
            img = torch.flip(img, dims=[-1])
            mask = torch.flip(mask, dims=[-1])
        
        return img, mask

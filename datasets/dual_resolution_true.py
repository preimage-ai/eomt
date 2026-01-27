"""
Dual-Resolution Dataset with True Mixed Sizes
Perspective: 512×512
ERP: 512×1024
"""

import os
import random
import zipfile
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

from datasets.transforms import get_transforms
from datasets.target_parser import TargetParser


class DualResolutionTrue(Dataset):
    """
    True dual-resolution dataset mixing perspective (512×512) and ERP (512×1024) images.
    Uses custom collate function to handle different sizes in same batch.
    """
    
    def __init__(
        self,
        perspective_path: str,
        erp_image_dir: str,
        erp_mask_dir: str,
        perspective_img_size: Tuple[int, int] = (512, 512),
        erp_img_size: Tuple[int, int] = (512, 1024),
        batch_size: int = 16,
        num_workers: int = 4,
        color_jitter_enabled: bool = True,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        perspective_ratio: float = 0.95,
        erp_augmentation_factor: int = 100,
    ):
        """
        Args:
            perspective_path: Path to ADE20K zip file
            erp_image_dir: Directory containing ERP images
            erp_mask_dir: Directory containing ERP masks
            perspective_img_size: Size for perspective images (H, W)
            erp_img_size: Size for ERP images (H, W)
            batch_size: Batch size
            num_workers: Number of workers
            color_jitter_enabled: Enable color jitter augmentation
            scale_range: Range for random scaling
            perspective_ratio: Ratio of perspective images in batch (0-1)
            erp_augmentation_factor: Number of augmented versions per ERP image
        """
        self.perspective_path = perspective_path
        self.erp_image_dir = Path(erp_image_dir)
        self.erp_mask_dir = Path(erp_mask_dir)
        self.perspective_img_size = perspective_img_size
        self.erp_img_size = erp_img_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.perspective_ratio = perspective_ratio
        self.erp_augmentation_factor = erp_augmentation_factor
        
        # Create datasets
        self.perspective_dataset = PerspectiveDataset(
            perspective_path,
            img_size=perspective_img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        
        self.erp_dataset = ERPDataset(
            erp_image_dir,
            erp_mask_dir,
            img_size=erp_img_size,
            augmentation_factor=erp_augmentation_factor,
            color_jitter_enabled=color_jitter_enabled,
        )
        
        # Mixed dataset
        self.mixed_dataset = MixedSizeDataset(
            self.perspective_dataset,
            self.erp_dataset,
            perspective_ratio=perspective_ratio,
        )
        
        print(f"Dual-Resolution True Dataset:")
        print(f"  Perspective: {len(self.perspective_dataset)} images at {perspective_img_size}")
        print(f"  ERP: {len(self.erp_dataset)} samples at {erp_img_size}")
        print(f"  Total: {len(self.mixed_dataset)} samples")
        print(f"  Perspective ratio: {perspective_ratio:.1%}")
    
    def __len__(self):
        return len(self.mixed_dataset)
    
    def __getitem__(self, idx):
        return self.mixed_dataset[idx]
    
    def train_dataloader(self):
        from torch.utils.data import DataLoader
        return DataLoader(
            self,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=dual_resolution_collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        from torch.utils.data import DataLoader
        # Validation uses ERP dataset only
        return DataLoader(
            self.erp_dataset,
            batch_size=1,  # Process one at a time for validation
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class PerspectiveDataset(Dataset):
    """ADE20K perspective images at 512×512"""
    
    def __init__(
        self,
        zip_path: str,
        img_size: Tuple[int, int] = (512, 512),
        color_jitter_enabled: bool = True,
        scale_range: Tuple[float, float] = (0.5, 2.0),
    ):
        self.zip_path = zip_path
        self.img_size = img_size
        self.transforms = get_transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        self.target_parser = TargetParser()
        
        # Load file list from zip
        self.pairs = []
        with zipfile.ZipFile(zip_path, 'r') as zf:
            all_files = zf.namelist()
            img_files = [f for f in all_files if '/images/training/' in f and f.endswith('.jpg')]
            
            for img_file in img_files:
                mask_file = img_file.replace('/images/training/', '/annotations/training/').replace('.jpg', '.png')
                if mask_file in all_files:
                    self.pairs.append((img_file, mask_file))
        
        print(f"Found {len(self.pairs)} perspective image pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img_path, mask_path = self.pairs[idx]
        
        with zipfile.ZipFile(self.zip_path, 'r') as zf:
            img = Image.open(zf.open(img_path)).convert('RGB')
            mask = Image.open(zf.open(mask_path))
        
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).long()
        
        if self.transforms:
            img, mask = self.transforms(img, mask)
        
        masks, labels, is_crowd = self.target_parser(mask)
        
        target = {
            "masks": torch.stack(masks) if masks else torch.zeros((0, *mask.shape[-2:])),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
            "size": torch.tensor(self.img_size),  # Mark size for collate function
        }
        
        return img, target


class ERPDataset(Dataset):
    """ERP images at 512×1024 with augmentation"""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: Tuple[int, int] = (512, 1024),
        augmentation_factor: int = 100,
        color_jitter_enabled: bool = True,
    ):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.img_size = img_size
        self.augmentation_factor = augmentation_factor
        self.color_jitter_enabled = color_jitter_enabled
        self.target_parser = TargetParser()
        
        # Find image-mask pairs
        self.pairs = []
        for img_path in sorted(self.image_dir.glob('*.png')):
            mask_path = self.mask_dir / img_path.name
            if mask_path.exists():
                self.pairs.append((img_path, mask_path))
        
        print(f"Found {len(self.pairs)} ERP image pairs")
    
    def __len__(self):
        return len(self.pairs) * self.augmentation_factor
    
    def __getitem__(self, idx):
        original_idx = idx % len(self.pairs)
        img_path, mask_path = self.pairs[original_idx]
        
        # Load at native resolution
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Resize to target size (512×1024)
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).long()
        
        # Apply ERP-specific augmentations
        img, mask = self._erp_augment(img, mask)
        
        masks, labels, is_crowd = self.target_parser(mask)
        
        target = {
            "masks": torch.stack(masks) if masks else torch.zeros((0, *mask.shape[-2:])),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
            "size": torch.tensor(self.img_size),  # Mark size for collate function
        }
        
        return img, target
    
    def _erp_augment(self, img, mask):
        """ERP-specific augmentations"""
        # Horizontal wrapping (panorama-specific)
        if random.random() < 0.8:
            shift = random.randint(0, img.shape[-1] - 1)
            img = torch.cat([img[:, :, shift:], img[:, :, :shift]], dim=-1)
            mask = torch.cat([mask[:, :, shift:], mask[:, :, :shift]], dim=-1)
        
        # Vertical shift (camera height variation)
        if random.random() < 0.5:
            shift = random.randint(-img.shape[-2] // 4, img.shape[-2] // 4)
            if shift > 0:
                img = torch.cat([img[:, shift:, :], img[:, :shift, :]], dim=-2)
                mask = torch.cat([mask[:, shift:, :], mask[:, :shift, :]], dim=-2)
            elif shift < 0:
                img = torch.cat([img[:, shift:, :], img[:, :shift, :]], dim=-2)
                mask = torch.cat([mask[:, shift:, :], mask[:, :shift, :]], dim=-2)
        
        # Horizontal flip
        if random.random() < 0.5:
            img = torch.flip(img, [-1])
            mask = torch.flip(mask, [-1])
        
        # Color jitter
        if self.color_jitter_enabled and random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
            img = TF.adjust_saturation(img, random.uniform(0.8, 1.2))
        
        return img, mask


class MixedSizeDataset(Dataset):
    """Mix perspective and ERP datasets with different sizes"""
    
    def __init__(
        self,
        perspective_dataset,
        erp_dataset,
        perspective_ratio: float,
    ):
        self.perspective_dataset = perspective_dataset
        self.erp_dataset = erp_dataset
        self.perspective_ratio = perspective_ratio
        self.length = max(len(perspective_dataset), len(erp_dataset))
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if random.random() < self.perspective_ratio:
            return self.perspective_dataset[idx % len(self.perspective_dataset)]
        else:
            return self.erp_dataset[idx % len(self.erp_dataset)]


def dual_resolution_collate_fn(batch):
    """
    Custom collate function to handle mixed-size images in same batch.
    Groups images by size and processes them separately.
    """
    # Separate by size
    size_512 = []
    size_1024 = []
    
    for img, target in batch:
        if img.shape[-1] == 512:  # 512×512
            size_512.append((img, target))
        elif img.shape[-1] == 1024:  # 512×1024
            size_1024.append((img, target))
    
    result = {}
    
    # Stack 512×512 images
    if size_512:
        result['imgs_512'] = torch.stack([x[0] for x in size_512])
        result['targets_512'] = [x[1] for x in size_512]
    else:
        result['imgs_512'] = None
        result['targets_512'] = []
    
    # Stack 512×1024 images
    if size_1024:
        result['imgs_1024'] = torch.stack([x[0] for x in size_1024])
        result['targets_1024'] = [x[1] for x in size_1024]
    else:
        result['imgs_1024'] = None
        result['targets_1024'] = []
    
    return result

"""
Full-Resolution Dual-Resolution Dataset
Perspective: 1024×1024 (padded from 512×512)
ERP: 1024×2048 (native resolution)
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


class DualResolutionFull(Dataset):
    """
    Full-resolution dual-resolution dataset.
    Perspective: Padded to 1024×1024
    ERP: Native 1024×2048
    """
    
    def __init__(
        self,
        perspective_path: str,
        erp_image_dir: str,
        erp_mask_dir: str,
        perspective_img_size: Tuple[int, int] = (1024, 1024),
        erp_img_size: Tuple[int, int] = (1024, 2048),
        batch_size: int = 4,
        num_workers: int = 4,
        color_jitter_enabled: bool = True,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        perspective_ratio: float = 0.95,
        erp_augmentation_factor: int = 50,
    ):
        """
        Args:
            perspective_path: Path to ADE20K zip file
            erp_image_dir: Directory containing ERP images
            erp_mask_dir: Directory containing ERP masks
            perspective_img_size: Size for perspective images (H, W)
            erp_img_size: Size for ERP images (H, W)
            batch_size: Batch size (smaller due to memory)
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
        self.perspective_dataset = PerspectiveDatasetFull(
            perspective_path,
            img_size=perspective_img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        
        self.erp_dataset = ERPDatasetFull(
            erp_image_dir,
            erp_mask_dir,
            img_size=erp_img_size,
            augmentation_factor=erp_augmentation_factor,
            color_jitter_enabled=color_jitter_enabled,
        )
        
        # Mixed dataset
        self.mixed_dataset = MixedSizeDatasetFull(
            self.perspective_dataset,
            self.erp_dataset,
            perspective_ratio=perspective_ratio,
        )
        
        print(f"Full-Resolution Dual-Resolution Dataset:")
        print(f"  Perspective: {len(self.perspective_dataset)} images at {perspective_img_size}")
        print(f"  ERP: {len(self.erp_dataset)} samples at {erp_img_size}")
        print(f"  Total: {len(self.mixed_dataset)} samples")
        print(f"  Perspective ratio: {perspective_ratio:.1%}")
        print(f"  Batch size: {batch_size} (optimized for 40GB GPU)")
    
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
            collate_fn=dual_resolution_full_collate_fn,
            pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
        )
    
    def val_dataloader(self):
        from torch.utils.data import DataLoader
        # Validation uses ERP dataset only
        return DataLoader(
            self.erp_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


class PerspectiveDatasetFull(Dataset):
    """ADE20K perspective images padded to 1024×1024"""
    
    def __init__(
        self,
        zip_path: str,
        img_size: Tuple[int, int] = (1024, 1024),
        color_jitter_enabled: bool = True,
        scale_range: Tuple[float, float] = (0.5, 2.0),
    ):
        self.zip_path = zip_path
        self.img_size = img_size
        self.transforms = get_transforms(
            img_size=(512, 512),  # Apply transforms at 512×512 first
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
        
        # Apply transforms at 512×512
        if self.transforms:
            img, mask = self.transforms(img, mask)
        
        # Pad to 1024×1024 (center padding)
        pad_h = self.img_size[0] - img.shape[1]
        pad_w = self.img_size[1] - img.shape[2]
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        img = torch.nn.functional.pad(img, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
        mask = torch.nn.functional.pad(mask, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=255)
        
        masks, labels, is_crowd = self.target_parser(mask)
        
        target = {
            "masks": torch.stack(masks) if masks else torch.zeros((0, *mask.shape[-2:])),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
            "size": torch.tensor(self.img_size),
        }
        
        return img, target


class ERPDatasetFull(Dataset):
    """ERP images at native 1024×2048 with augmentation"""
    
    def __init__(
        self,
        image_dir: str,
        mask_dir: str,
        img_size: Tuple[int, int] = (1024, 2048),
        augmentation_factor: int = 50,
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
        
        # Resize to target size if needed
        if img.size != (self.img_size[1], self.img_size[0]):
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
            "size": torch.tensor(self.img_size),
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


class MixedSizeDatasetFull(Dataset):
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


def dual_resolution_full_collate_fn(batch):
    """
    Custom collate function for full-resolution mixed-size batches.
    Groups 1024×1024 and 1024×2048 images separately.
    """
    # Separate by size
    size_1024 = []
    size_2048 = []
    
    for img, target in batch:
        if img.shape[-1] == 1024:  # 1024×1024
            size_1024.append((img, target))
        elif img.shape[-1] == 2048:  # 1024×2048
            size_2048.append((img, target))
    
    result = {}
    
    # Stack 1024×1024 images
    if size_1024:
        result['imgs_1024'] = torch.stack([x[0] for x in size_1024])
        result['targets_1024'] = [x[1] for x in size_1024]
    else:
        result['imgs_1024'] = None
        result['targets_1024'] = []
    
    # Stack 1024×2048 images
    if size_2048:
        result['imgs_2048'] = torch.stack([x[0] for x in size_2048])
        result['targets_2048'] = [x[1] for x in size_2048]
    else:
        result['imgs_2048'] = None
        result['targets_2048'] = []
    
    return result

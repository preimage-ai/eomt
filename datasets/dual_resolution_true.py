# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

"""
True Dual-Resolution Dataset: 512×512 perspective + 512×1024 ERP
Uses custom collate function to handle mixed sizes in same batch
"""

from pathlib import Path
from typing import Union, Tuple
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, Dataset as TorchDataset
from PIL import Image
import torchvision.transforms.functional as TF

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms

CLASS_MAPPING = {i: i - 1 for i in range(1, 167)}


class DualResolutionTrue(LightningDataModule):
    """
    True dual-resolution training with 512×512 perspective and 512×1024 ERP.
    Mixed-size batches handled by custom collate function.
    """
    
    def __init__(
        self,
        perspective_path: str,
        erp_image_dir: str,
        erp_mask_dir: str,
        num_workers: int = 4,
        batch_size: int = 16,
        perspective_img_size: Tuple[int, int] = (512, 512),
        erp_img_size: Tuple[int, int] = (512, 1024),
        num_classes: int = 166,
        color_jitter_enabled: bool = True,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
        perspective_ratio: float = 0.95,
        erp_augmentation_factor: int = 100,
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
        self.erp_img_size = erp_img_size
        self.perspective_ratio = perspective_ratio
        self.erp_augmentation_factor = erp_augmentation_factor
        self.color_jitter_enabled = color_jitter_enabled
        self.scale_range = scale_range
        
        print(f"\n{'='*70}")
        print(f"True Dual-Resolution Training")
        print(f"{'='*70}")
        print(f"Perspective: {perspective_img_size}")
        print(f"ERP: {erp_img_size}")
        print(f"Batch size: {batch_size}")
        print(f"Perspective ratio: {perspective_ratio:.1%}")
        print(f"{'='*70}\n")
        
        self.save_hyperparameters(ignore=["_class_path"])
        
        # Create transforms
        self.perspective_transforms = Transforms(
            img_size=perspective_img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
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
        # Load perspective dataset
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "zip_path": Path(self.perspective_path, "ADEChallengeData2016.zip"),
            "target_zip_path": Path(self.perspective_path, "ADEChallengeData2016.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        
        perspective_train = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/training"),
            target_folder_path_in_zip=Path("./ADEChallengeData2016/annotations/training"),
            transforms=self.perspective_transforms,
            **dataset_kwargs,
        )
        
        perspective_val = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/validation"),
            target_folder_path_in_zip=Path("./ADEChallengeData2016/annotations/validation"),
            transforms=None,
            **dataset_kwargs,
        )
        
        # Load ERP images
        erp_images = sorted(list(self.erp_image_dir.glob("*.jpg")) + list(self.erp_image_dir.glob("*.png")))
        if len(erp_images) == 0:
            raise ValueError(f"No ERP images found in {self.erp_image_dir}")
        
        erp_masks = [self.erp_mask_dir / f"{img.stem}_mask_ids.png" for img in erp_images]
        erp_pairs = [(img, mask) for img, mask in zip(erp_images, erp_masks) if mask.exists()]
        
        if len(erp_pairs) == 0:
            raise ValueError(f"No ERP mask pairs found in {self.erp_mask_dir}")
        
        print(f"Found {len(perspective_train)} perspective training images")
        print(f"Found {len(perspective_val)} perspective validation images")
        print(f"Found {len(erp_pairs)} ERP images")
        
        # Split ERP
        val_split = max(1, len(erp_pairs) // 10)
        erp_train_pairs = erp_pairs[val_split:]
        erp_val_pairs = erp_pairs[:val_split]
        
        # Create ERP datasets
        erp_train = ERPDatasetTrue(
            erp_train_pairs,
            img_size=self.erp_img_size,
            augmentation_factor=self.erp_augmentation_factor,
            target_parser=self.target_parser,
            color_jitter_enabled=self.color_jitter_enabled,
        )
        
        erp_val = ERPDatasetTrue(
            erp_val_pairs,
            img_size=self.erp_img_size,
            augmentation_factor=1,
            target_parser=self.target_parser,
            color_jitter_enabled=False,
        )
        
        # Create mixed datasets
        self.train_dataset = MixedSizeDataset(perspective_train, erp_train, self.perspective_ratio)
        self.val_dataset = MixedSizeDataset(perspective_val, erp_val, self.perspective_ratio)
        
        # Store ERP validation dataset separately for validation dataloader
        self.erp_val_dataset = erp_val
        
        return self

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            drop_last=True,
            collate_fn=dual_resolution_collate_fn,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self):
        # Validation uses only ERP dataset to avoid mixed-size issues
        return DataLoader(
            self.erp_val_dataset,
            shuffle=False,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )


class ERPDatasetTrue(TorchDataset):
    """ERP dataset at 512×1024"""
    
    def __init__(self, pairs, img_size, augmentation_factor, target_parser, color_jitter_enabled):
        self.pairs = pairs
        self.img_size = img_size
        self.augmentation_factor = augmentation_factor
        self.target_parser = target_parser
        self.color_jitter_enabled = color_jitter_enabled
    
    def __len__(self):
        return len(self.pairs) * self.augmentation_factor
    
    def __getitem__(self, idx):
        original_idx = idx % len(self.pairs)
        img_path, mask_path = self.pairs[original_idx]
        
        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        # Resize to target
        img = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        mask = mask.resize((self.img_size[1], self.img_size[0]), Image.NEAREST)
        
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(np.array(mask)).unsqueeze(0).long()
        
        # ERP augmentation
        if random.random() < 0.8:
            shift = random.randint(0, img.shape[-1] - 1)
            img = torch.cat([img[:, :, shift:], img[:, :, :shift]], dim=-1)
            mask = torch.cat([mask[:, :, shift:], mask[:, :, :shift]], dim=-1)
        
        if random.random() < 0.5:
            img = torch.flip(img, [-1])
            mask = torch.flip(mask, [-1])
        
        if self.color_jitter_enabled and random.random() < 0.5:
            img = TF.adjust_brightness(img, random.uniform(0.8, 1.2))
            img = TF.adjust_contrast(img, random.uniform(0.8, 1.2))
        
        masks, labels, is_crowd = self.target_parser(mask)
        
        target = {
            "masks": torch.stack(masks) if masks else torch.zeros((0, *mask.shape[-2:])),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
            "is_rooftop": True,  # All ERP data is rooftop category
        }
        
        return img, target


class MixedSizeDataset(TorchDataset):
    """Mix perspective and ERP datasets"""
    
    def __init__(self, perspective_dataset, erp_dataset, perspective_ratio):
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
    """Custom collate for mixed-size batches"""
    size_512 = []
    size_1024 = []
    
    for img, target in batch:
        if img.shape[-1] == 512:
            size_512.append((img, target))
        elif img.shape[-1] == 1024:
            size_1024.append((img, target))
    
    result = {}
    
    if size_512:
        result['imgs_512'] = torch.stack([x[0] for x in size_512])
        result['targets_512'] = [x[1] for x in size_512]
    else:
        result['imgs_512'] = None
        result['targets_512'] = []
    
    if size_1024:
        result['imgs_1024'] = torch.stack([x[0] for x in size_1024])
        result['targets_1024'] = [x[1] for x in size_1024]
    else:
        result['imgs_1024'] = None
        result['targets_1024'] = []
    
    return result

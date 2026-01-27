# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union, Tuple
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from PIL import Image

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms

CLASS_MAPPING = {i: i - 1 for i in range(1, 167)}


class ERPFineTuning(LightningDataModule):
    """
    Dataset for fine-tuning on limited ERP images with aggressive augmentation.
    
    Strategy:
    - Heavy augmentation on 20 ERP images
    - Optional: Mix with perspective images at low ratio
    """
    def __init__(
        self,
        erp_image_dir: str,
        erp_mask_dir: str,
        perspective_path: str = None,  # Optional ADE20K path for mixing
        num_workers: int = 4,
        batch_size: int = 4,  # Small batch for limited data
        erp_img_size: Tuple[int, int] = (512, 1024),
        num_classes: int = 166,
        color_jitter_enabled: bool = True,
        scale_range: Tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
        perspective_mix_ratio: float = 0.0,  # 0.0 = pure ERP, 0.8 = 80% perspective
        erp_augmentation_factor: int = 50,  # Repeat ERP samples 50x with different augmentations
    ) -> None:
        super().__init__(
            path=erp_image_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=erp_img_size,
            check_empty_targets=check_empty_targets,
        )
        
        self.erp_image_dir = Path(erp_image_dir)
        self.erp_mask_dir = Path(erp_mask_dir)
        self.perspective_path = perspective_path
        self.perspective_mix_ratio = perspective_mix_ratio
        self.erp_augmentation_factor = erp_augmentation_factor
        
        print(f"ERP Fine-tuning Dataset:")
        print(f"  - ERP images: {erp_image_dir}")
        print(f"  - ERP size: {erp_img_size}")
        print(f"  - Augmentation factor: {erp_augmentation_factor}x")
        print(f"  - Perspective mix ratio: {perspective_mix_ratio}")
        
        self.save_hyperparameters(ignore=["_class_path"])

        # Aggressive augmentation for ERP
        self.erp_transforms = ERPAugmentedTransforms(
            img_size=erp_img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
            horizontal_wrap=True,  # Panorama-specific
            vertical_shift=True,
        )
        
        # Standard transforms for perspective (if used)
        if perspective_path:
            self.perspective_transforms = Transforms(
                img_size=(512, 512),
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
        # Load ERP images
        erp_images = sorted(list(self.erp_image_dir.glob("*.jpg")) + 
                           list(self.erp_image_dir.glob("*.png")))
        erp_masks = [self.erp_mask_dir / f"{img.stem}.png" for img in erp_images]
        
        # Filter valid pairs
        valid_pairs = [(img, mask) for img, mask in zip(erp_images, erp_masks) if mask.exists()]
        
        print(f"Found {len(valid_pairs)} ERP image-mask pairs")
        
        if len(valid_pairs) == 0:
            raise ValueError(f"No ERP images found in {self.erp_image_dir}")
        
        # Create augmented ERP dataset
        self.erp_train_dataset = ERPAugmentedDataset(
            valid_pairs,
            transforms=self.erp_transforms,
            augmentation_factor=self.erp_augmentation_factor,
            target_parser=self.target_parser,
        )
        
        # Optionally load perspective dataset for mixing
        if self.perspective_path and self.perspective_mix_ratio > 0:
            dataset_kwargs = {
                "img_suffix": ".jpg",
                "target_suffix": ".png",
                "zip_path": Path(self.perspective_path, "ADEChallengeData2016.zip"),
                "target_zip_path": Path(self.perspective_path, "ADEChallengeData2016.zip"),
                "target_parser": self.target_parser,
                "check_empty_targets": self.check_empty_targets,
            }
            
            self.perspective_train_dataset = Dataset(
                img_folder_path_in_zip=Path("./ADEChallengeData2016/images/training"),
                target_folder_path_in_zip=Path("./ADEChallengeData2016/annotations/training"),
                transforms=self.perspective_transforms,
                **dataset_kwargs,
            )
        else:
            self.perspective_train_dataset = None
        
        # Validation uses first 2-3 ERP images (or separate val set if available)
        val_split = max(1, len(valid_pairs) // 10)  # 10% for validation
        self.val_dataset = ERPAugmentedDataset(
            valid_pairs[:val_split],
            transforms=None,  # No augmentation for validation
            augmentation_factor=1,
            target_parser=self.target_parser,
        )

        return self

    def train_dataloader(self):
        if self.perspective_train_dataset and self.perspective_mix_ratio > 0:
            # Mix ERP and perspective
            class MixedDataset(torch.utils.data.Dataset):
                def __init__(self, erp_dataset, perspective_dataset, perspective_ratio):
                    self.erp_dataset = erp_dataset
                    self.perspective_dataset = perspective_dataset
                    self.perspective_ratio = perspective_ratio
                    self.length = len(erp_dataset)
                
                def __len__(self):
                    return self.length
                
                def __getitem__(self, idx):
                    if random.random() < self.perspective_ratio:
                        idx = random.randint(0, len(self.perspective_dataset) - 1)
                        return self.perspective_dataset[idx]
                    else:
                        return self.erp_dataset[idx]
            
            dataset = MixedDataset(
                self.erp_train_dataset,
                self.perspective_train_dataset,
                self.perspective_mix_ratio
            )
        else:
            dataset = self.erp_train_dataset
        
        return DataLoader(
            dataset,
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
    """Aggressive augmentation for ERP images"""
    
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
        if self.horizontal_wrap and random.random() < 0.7:
            # Horizontal wrapping (panoramas are cyclic)
            shift = random.randint(0, img.shape[-1])
            img = torch.roll(img, shift, dims=-1)
            mask = torch.roll(mask, shift, dims=-1)
        
        if self.vertical_shift and random.random() < 0.5:
            # Vertical shift (simulate different camera heights)
            shift = random.randint(-img.shape[-2] // 4, img.shape[-2] // 4)
            img = torch.roll(img, shift, dims=-2)
            mask = torch.roll(mask, shift, dims=-2)
        
        # Additional color augmentation for ERP
        if random.random() < 0.5:
            # Brightness adjustment
            factor = random.uniform(0.8, 1.2)
            img = torch.clamp(img * factor, 0, 1)
        
        return img, mask

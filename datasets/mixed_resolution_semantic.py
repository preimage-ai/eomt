# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from pathlib import Path
from typing import Union, List, Tuple
import random
import torch
from torch.utils.data import DataLoader

from datasets.lightning_data_module import LightningDataModule
from datasets.dataset import Dataset
from datasets.transforms import Transforms

CLASS_MAPPING = {i: i - 1 for i in range(1, 167)}


class MixedResolutionSemantic(LightningDataModule):
    """
    Dataset that supports training on mixed image resolutions:
    - Perspective images: (512, 512)
    - ERP/Panorama images: (512, 1024)
    
    Randomly samples between the two resolutions during training.
    """
    def __init__(
        self,
        path,
        num_workers: int = 4,
        batch_size: int = 8,
        perspective_img_size: Tuple[int, int] = (512, 512),
        erp_img_size: Tuple[int, int] = (512, 1024),
        num_classes: int = 166,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        check_empty_targets=True,
        perspective_ratio: float = 0.5,  # Ratio of perspective vs ERP images
    ) -> None:
        # Use ERP size as default for validation
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            num_classes=num_classes,
            img_size=erp_img_size,
            check_empty_targets=check_empty_targets,
        )
        
        self.perspective_img_size = perspective_img_size
        self.erp_img_size = erp_img_size
        self.perspective_ratio = perspective_ratio
        
        print(f"Mixed Resolution Training:")
        print(f"  - Perspective: {perspective_img_size}, ratio: {perspective_ratio}")
        print(f"  - ERP: {erp_img_size}, ratio: {1-perspective_ratio}")
        print(f"  - Num classes: {num_classes}")
        
        self.save_hyperparameters(ignore=["_class_path"])

        # Create transforms for both resolutions
        self.perspective_transforms = Transforms(
            img_size=perspective_img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )
        
        self.erp_transforms = Transforms(
            img_size=erp_img_size,
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
        dataset_kwargs = {
            "img_suffix": ".jpg",
            "target_suffix": ".png",
            "zip_path": Path(self.path, "ADEChallengeData2016.zip"),
            "target_zip_path": Path(self.path, "ADEChallengeData2016.zip"),
            "target_parser": self.target_parser,
            "check_empty_targets": self.check_empty_targets,
        }
        
        # Create perspective dataset
        self.perspective_train_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/training"),
            target_folder_path_in_zip=Path(
                "./ADEChallengeData2016/annotations/training"
            ),
            transforms=self.perspective_transforms,
            **dataset_kwargs,
        )
        
        # Create ERP dataset
        self.erp_train_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/training"),
            target_folder_path_in_zip=Path(
                "./ADEChallengeData2016/annotations/training"
            ),
            transforms=self.erp_transforms,
            **dataset_kwargs,
        )
        
        # Validation uses ERP resolution
        self.val_dataset = Dataset(
            img_folder_path_in_zip=Path("./ADEChallengeData2016/images/validation"),
            target_folder_path_in_zip=Path(
                "./ADEChallengeData2016/annotations/validation"
            ),
            transforms=None,
            **dataset_kwargs,
        )

        return self

    def train_dataloader(self):
        """
        Custom dataloader that randomly samples from perspective or ERP datasets
        """
        # Create a mixed dataset wrapper
        class MixedDataset(torch.utils.data.Dataset):
            def __init__(self, perspective_dataset, erp_dataset, perspective_ratio):
                self.perspective_dataset = perspective_dataset
                self.erp_dataset = erp_dataset
                self.perspective_ratio = perspective_ratio
                # Use the longer dataset length
                self.length = max(len(perspective_dataset), len(erp_dataset))
            
            def __len__(self):
                return self.length
            
            def __getitem__(self, idx):
                # Randomly choose between perspective and ERP
                if random.random() < self.perspective_ratio:
                    # Wrap index if needed
                    idx = idx % len(self.perspective_dataset)
                    return self.perspective_dataset[idx]
                else:
                    idx = idx % len(self.erp_dataset)
                    return self.erp_dataset[idx]
        
        mixed_dataset = MixedDataset(
            self.perspective_train_dataset,
            self.erp_train_dataset,
            self.perspective_ratio
        )
        
        return DataLoader(
            mixed_dataset,
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

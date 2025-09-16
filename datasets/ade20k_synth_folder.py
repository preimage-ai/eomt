# ---------------------------------------------------------------
# ADE20K Synthetic (Folder-based) Semantic Segmentation DataModule
# Reads directly from folders (no zips) and performs a 90/10 split.
#
# Expected structure under <root>:
#   images_resampled/*.jpg
#   annotations_filtered_resampled/*.png
#
# Matches files by identical names (e.g., 000123.jpg in both dirs).
# ---------------------------------------------------------------

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F
from PIL import Image

# Reuse your project’s utilities
from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms

# Map ADE20K semantic ids (1..150) -> 0-based indices (0..149)
CLASS_MAPPING = {i: i - 1 for i in range(1, 151)}


def default_target_parser(
    target: tv_tensors.Mask,
) -> tuple[list[torch.Tensor], list[int], list[bool]]:
    """
    Convert a single-channel semantic label map to binary masks per present class.

    Args:
        target: tv_tensors.Mask (H, W) with integer class ids per pixel.

    Returns:
        masks  : list[BoolTensor(H, W)] one per present class id
        labels : list[int] class indices mapped to [0..num_classes-1]
        is_crowd: list[bool] (always False here)
    """
    # Ensure long dtype for unique()
    t = target.to(dtype=torch.int64)
    unique_ids = torch.unique(t)

    masks, labels = [], []
    for cls_id in unique_ids.tolist():
        if cls_id in CLASS_MAPPING:
            masks.append(t == cls_id)
            labels.append(CLASS_MAPPING[cls_id])

    return masks, labels, [False] * len(masks)


@dataclass(frozen=True)
class _Item:
    img_path: Path
    mask_path: Path


class _FolderDataset(torch.utils.data.Dataset):
    """
    Folder-based dataset that:
      - pairs same-named PNGs between images_resampled/ and annotations_filtered_resampled/
      - optionally skips pure-background masks
      - returns (tv_tensors.Image, target_dict) where target_dict has:
        {
          "masks": tv_tensors.Mask[B, H, W] (boolean),
          "labels": LongTensor[B],
          "is_crowd": BoolTensor[B]
        }
    """

    def __init__(
        self,
        root: Path,
        images_dir: str = "images_resampled",
        masks_dir: str = "annotations_filtered_resampled",
        img_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        transforms: Optional[Callable] = None,
        target_parser: Callable[[tv_tensors.Mask], tuple[list[torch.Tensor], list[int], list[bool]]] = default_target_parser,
        check_empty_targets: bool = True,
    ) -> None:
        super().__init__()
        self.root = Path(root)
        self.images_dir = self.root / images_dir
        self.masks_dir = self.root / masks_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.transforms = transforms
        self.target_parser = target_parser
        self.check_empty_targets = check_empty_targets

        if not self.images_dir.is_dir():
            raise FileNotFoundError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.is_dir():
            raise FileNotFoundError(f"Masks directory not found: {self.masks_dir}")

        # Build file pairs by exact same filename (stem + suffix) present in both dirs
        images = sorted(self.images_dir.glob(f"*{self.img_suffix}"))
        candidates: list[_Item] = []
        for img_path in images:
            mask_path = self.masks_dir / img_path.name.replace(self.img_suffix, self.mask_suffix)
            if not mask_path.exists():
                continue

            if self.check_empty_targets:
                # Skip pure-background masks (min==max==0)
                with Image.open(mask_path) as m:
                    extrema = m.getextrema()
                    if (
                        isinstance(extrema, tuple)
                        and len(extrema) == 2
                        and extrema[0] == extrema[1] == 0
                    ):
                        continue

            candidates.append(_Item(img_path=img_path, mask_path=mask_path))

        if not candidates:
            raise RuntimeError(
                "No valid image/mask pairs found.\n"
                f"  Images: {self.images_dir}\n"
                f"  Masks : {self.masks_dir}\n"
                f"(Looked for *{self.img_suffix} with matching *{self.mask_suffix})"
            )

        self._items = candidates

    def __len__(self) -> int:
        return len(self._items)

    def __getitem__(self, index: int):
        item = self._items[index]

        # Load image (RGB)
        img = tv_tensors.Image(Image.open(item.img_path).convert("RGB"))

        # Load mask as integer label map (L)
        mask = tv_tensors.Mask(Image.open(item.mask_path).convert("L"))

        # Ensure same spatial size (use NEAREST for masks)
        img_size = F.get_size(img)  # (H, W)
        if F.get_size(mask) != img_size:
            mask = F.resize(mask, img_size, interpolation=InterpolationMode.NEAREST)

        # Parse to instance-like target dict: binary masks per present class
        masks, labels, is_crowd = self.target_parser(mask)

        if len(masks) == 0:
            # If a sample ends up empty after any transforms (rare), attempt a fallback.
            t = mask.to(dtype=torch.int64)
            present = [cid for cid in torch.unique(t).tolist() if cid in CLASS_MAPPING]
            if not present:
                raise RuntimeError(f"Mask at {item.mask_path} has no valid classes.")
            cid = present[0]
            masks = [t == cid]
            labels = [CLASS_MAPPING[cid]]
            is_crowd = [False]

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks, dim=0)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target


class ADE20KSynthSemantic(LightningDataModule):
    """
    Lightning DataModule for ADE20K Synthetic (semantic):
      - reads directly from folders (no zips)
      - does an internal 90/10 train/val split (configurable)
      - train uses your Transforms (aug); val is no-aug
    """

    def __init__(
        self,
        path: str | Path,
        num_workers: int = 4,
        batch_size: int = 8,
        img_size: tuple[int, int] = (512, 512),
        num_classes: int = 150,
        color_jitter_enabled: bool = True,
        scale_range: tuple[float, float] = (0.5, 2.0),
        check_empty_targets: bool = True,
        images_dir: str = "images",
        masks_dir: str = "annotations_filtered",
        img_suffix: str = ".jpg",
        mask_suffix: str = ".png",
        val_ratio: float = 0.1,
        split_seed: int = 42,
    ) -> None:
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            num_classes=num_classes,
            check_empty_targets=check_empty_targets,
        )
        self.save_hyperparameters(ignore=["_class_path"])

        # Train-time transforms (keep consistent with your project)
        self.transforms = Transforms(
            img_size=img_size,
            color_jitter_enabled=color_jitter_enabled,
            scale_range=scale_range,
        )

        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix

        if not (0.0 < float(val_ratio) < 1.0):
            raise ValueError("val_ratio must be in (0, 1).")
        self.val_ratio = float(val_ratio)
        self.split_seed = int(split_seed)

        self._train_ds = None
        self._val_ds = None

    @staticmethod
    def target_parser(target: tv_tensors.Mask, **kwargs):
        # Wrap the default parser (kept static to mirror your style)
        return default_target_parser(target)

    def _make_full_dataset(self, with_transforms: bool) -> _FolderDataset:
        return _FolderDataset(
            root=Path(self.path),
            images_dir=self.images_dir,
            masks_dir=self.masks_dir,
            img_suffix=self.img_suffix,
            mask_suffix=self.mask_suffix,
            transforms=self.transforms if with_transforms else None,
            target_parser=self.target_parser,
            check_empty_targets=self.check_empty_targets,
        )

    def setup(self, stage: Optional[str] = None) -> "ADE20KSynthSemantic":
        # Build once then split deterministically
        full_ds = self._make_full_dataset(with_transforms=True)   # train (aug)
        n = len(full_ds)
        n_val = max(1, int(math.floor(self.val_ratio * n)))
        n_train = n - n_val

        g = torch.Generator()
        g.manual_seed(self.split_seed)
        train_indices, val_indices = torch.utils.data.random_split(
            range(n), lengths=[n_train, n_val], generator=g
        )

        # For val: same underlying data, but no transforms (no aug)
        full_ds_no_aug = self._make_full_dataset(with_transforms=False)

        self._train_ds = torch.utils.data.Subset(full_ds, train_indices)
        self._val_ds = torch.utils.data.Subset(full_ds_no_aug, val_indices)
        return self

    # Dataloaders use LightningDataModule's collates/kwargs
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self._train_ds,
            shuffle=True,
            drop_last=True,
            collate_fn=self.train_collate,
            **self.dataloader_kwargs,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self._val_ds,
            shuffle=False,
            collate_fn=self.eval_collate,
            **self.dataloader_kwargs,
        )

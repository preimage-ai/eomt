# Combined ADE20K Synthetic Round-Robin + Folder-Based Semantic Segmentation DataModule
# (CSV-driven unified loader)

"""
This script unifies both datasets:
1. Multi-source round-robin edited scenes (from CSV)
2. Folder-based resampled ADE20K images (images_resampled + annotations_filtered_resampled)

The unified CSV contains rows with a new column "mode":
    - mode == "multi" → multi-source edited dataset
    - mode == "folder" → folder-based dataset

For folder rows:
    - scene_name: unique key (e.g., "ade_train_00000001")
    - sources: "images_resampled" (or blank)
    - mask_sources: "annotations_filtered_resampled" (or blank)
    - full_img_path: optional direct path

The Dataloader will load both datasets in one unified Dataset.
"""

from __future__ import annotations
import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional
from collections import defaultdict

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import tv_tensors
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import functional as F
from PIL import Image

from datasets.lightning_data_module import LightningDataModule
from datasets.transforms import Transforms

# Map ADE20K semantic ids
CLASS_MAPPING = {i: i - 1 for i in range(1, 198)}

# ---------- Target Parser ----------
def default_target_parser(target: tv_tensors.Mask):
    t = target.to(dtype=torch.int64)
    unique_ids = torch.unique(t)

    masks, labels = [], []
    for cls_id in unique_ids.tolist():
        if cls_id in CLASS_MAPPING:
            masks.append(t == cls_id)
            labels.append(CLASS_MAPPING[cls_id])

    return masks, labels, [False] * len(masks)

# ---------- Dataset Item Types ----------
@dataclass(frozen=True)
class MultiSourceItem:
    scene_name: str
    source_folders: list[str]
    root: Path
    img_suffix: str
    mask_suffix: str

    def get_paths(self, idx: int):
        folder = self.source_folders[idx % len(self.source_folders)]
        base = self.root / folder / "vlm_modified"
        return (
            base / f"{self.scene_name}{self.img_suffix}",
            base / f"{self.scene_name}{self.mask_suffix}",
        )

@dataclass(frozen=True)
class FolderItem:
    img_path: Path
    mask_path: Path

# ---------- Unified Dataset ----------
class UnifiedADE20K(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Path,
        csv_path: Path,
        transforms: Optional[Callable],
        target_parser: Callable,
        img_suffix: str,
        mask_suffix: str,
        check_empty_targets: bool,
    ):
        self.root = root
        self.transforms = transforms
        self.target_parser = target_parser
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.check_empty_targets = check_empty_targets
        self._access_counters = defaultdict(int)

        self.multi_items: list[MultiSourceItem] = []
        self.folder_items: list[FolderItem] = []
        self._parse_csv(csv_path)

        self.items = self.multi_items + self.folder_items
        self.debug = False

    # ---------- Parse CSV ----------
    def _parse_csv(self, csv_path: Path):
        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():
            mode = str(row.get("mode", "multi")).strip()

            if mode == "multi":
                sn = str(row["scene_name"]).strip()
                sources = [s.strip() for s in str(row["sources"]).split(',') if s.strip()]
                sources_ = []
                class_names = [c.strip() for c in str(row["class_names"]).split(';') if c.strip]
                # need fixing, we need to reference all versions of a class, maybe condition on class_ids

                for itm_name in ["fire_extinguisher", "extinguisher_cylinder", "cable_tray", "wire_tray", "ladder_tray", "electrical_raceway",
                                "radiator", "heating_radiator", "convector", "electrical_panel", "distribution_panel", "DB_board", "switchgear_panel", "control_board",
                                "hvac_duct", "air_duct", "fresh_air_duct", "supply_duct", "return_duct"]:
                    if itm_name in class_names:
                        sources.append([class_names.index(itm_name)])
                    else:
                        continue
                if len(sources_) == 0:
                    continue

                self.multi_items.append(
                    MultiSourceItem(
                        scene_name=sn,
                        source_folders=sources,
                        root=Path("/home/shifu/nas/jd-dainty/qwen-image-edit/images_train"),
                        img_suffix=self.img_suffix,
                        mask_suffix=self.mask_suffix,
                    )
                )

            elif mode == "folder":
                img = Path(row["full_img_path"]) if "full_img_path" in row else None
                if img is None or not img.exists():
                    # build from folders
                    sn = str(row["scene_name"]).strip()
                    img = self.root / "ade20k_images" / f"{sn}.jpg"
                    mask = self.root / "ade20k_mask" / f"{sn}.png"
                else:
                    mask = Path(row["full_mask_path"])

                if img.exists() and mask.exists():
                    self.folder_items.append(FolderItem(img_path=img, mask_path=mask))

    def __len__(self) -> int:
        return len(self.items)

    # ---------- Data Loading ----------
    def __getitem__(self, index):
        item = self.items[index]

        if isinstance(item, MultiSourceItem):
            rr = self._access_counters[item.scene_name]
            self._access_counters[item.scene_name] += 1
            img_path, mask_path = item.get_paths(rr)

            if self.debug:
                print(f"[MULTI] scene={item.scene_name}  "
                    f"rr_idx={rr % len(item.source_folders)}  "
                    f"folder={item.source_folders[rr % len(item.source_folders)]}\n"
                    f"      img={img_path}\n"
                    f"      mask={mask_path}")
        else:
            img_path, mask_path = item.img_path, item.mask_path
            if self.debug:
                print(f"[FOLDER] scene={item.img_path.stem}\n"
                    f"         img={img_path}\n"
                    f"         mask={mask_path}")

        # load
        if not img_path.exists():
            raise FileNotFoundError(f"[IMAGE MISSING] Cannot continue: {img_path}")

        # --- LOAD IMAGE ---
        try:
            img = tv_tensors.Image(Image.open(img_path).convert("RGB"))
        except Exception as e:
            return self.__getitem__((index + 1) % len(self))
            

        # --- CHECK MASK EXISTS (MASK OPTIONAL BUT SHOULD SKIP IF MISSING) ---
        if not mask_path.exists():
            return self.__getitem__((index + 1) % len(self))

        # --- LOAD MASK (SKIP IF BAD) ---
        try:
            mask = tv_tensors.Mask(Image.open(mask_path).convert("L"))
        except Exception as e:
            print(f"[SKIP] Mask corrupt → skipping sample\n MASK: {mask_path}\n Error: {e}")
            return self.__getitem__((index + 1) % len(self))

        img_size = F.get_size(img)
        if F.get_size(mask) != img_size:
            mask = F.resize(mask, img_size, interpolation=InterpolationMode.NEAREST)

        masks, labels, is_crowd = self.target_parser(mask)
        if len(masks) == 0:
            t = mask.to(dtype=torch.int64)
            present = [cid for cid in torch.unique(t).tolist() if cid in CLASS_MAPPING]
            if not present:
                return self.__getitem__((index + 1) % len(self))
            cid = present[0]
            masks = [t == cid]
            labels = [CLASS_MAPPING[cid]]
            is_crowd = [False]

        target = {
            "masks": tv_tensors.Mask(torch.stack(masks, 0)),
            "labels": torch.tensor(labels, dtype=torch.long),
            "is_crowd": torch.tensor(is_crowd, dtype=torch.bool),
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

# ---------- Lightning DataModule ----------
class UnifiedADE20KDataModule(LightningDataModule):
    def __init__(
        self,
        path: str | Path,
        csv_path: str | Path,
        num_workers=4,
        batch_size=8,
        img_size=(512, 512),
        num_classes=165,
        color_jitter_enabled=True,
        scale_range=(0.5, 2.0),
        val_ratio=0.1,
        split_seed=42,
        img_suffix="_edited.jpg",
        mask_suffix="_edited.png",
        check_empty_targets=True,
    ):
        super().__init__(
            path=path,
            batch_size=batch_size,
            num_workers=num_workers,
            img_size=img_size,
            num_classes=num_classes,
            check_empty_targets=check_empty_targets,
        )
        self.csv_path = Path(csv_path)
        self.transforms = Transforms(img_size, color_jitter_enabled, scale_range)
        self.val_ratio = val_ratio
        self.split_seed = split_seed
        self.img_suffix = img_suffix
        self.mask_suffix = mask_suffix
        self.check_empty_targets = check_empty_targets

    @staticmethod
    def target_parser(target: tv_tensors.Mask):
        return default_target_parser(target)

    def setup(self, stage=None):
        ds = UnifiedADE20K(
            root=Path(self.path),
            csv_path=self.csv_path,
            transforms=self.transforms,
            target_parser=self.target_parser,
            img_suffix=self.img_suffix,
            mask_suffix=self.mask_suffix,
            check_empty_targets=self.check_empty_targets,
        )

        n = len(ds)
        n_val = max(1, int(math.floor(self.val_ratio * n)))
        n_train = n - n_val

        g = torch.Generator().manual_seed(self.split_seed)
        train_idx, val_idx = torch.utils.data.random_split(range(n), [n_train, n_val], generator=g)

        # val without aug
        ds_no_aug = UnifiedADE20K(
            root=Path(self.path),
            csv_path=self.csv_path,
            transforms=None,
            target_parser=self.target_parser,
            img_suffix=self.img_suffix,
            mask_suffix=self.mask_suffix,
            check_empty_targets=self.check_empty_targets,
        )

        self._train_ds = torch.utils.data.Subset(ds, train_idx)
        self._val_ds = torch.utils.data.Subset(ds_no_aug, val_idx)
        return self

    def train_dataloader(self):
        return DataLoader(self._train_ds, shuffle=True, drop_last=True,
                          collate_fn=self.train_collate, **self.dataloader_kwargs)

    def val_dataloader(self):
        return DataLoader(self._val_ds, shuffle=False,
                          collate_fn=self.eval_collate, **self.dataloader_kwargs)

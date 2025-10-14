# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from Detectron2 by Facebook, Inc. and its affiliates,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import math

import torch
import torch.nn.functional as Fnn
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import wrap, TVTensor
from torch import nn, Tensor
from typing import Any, Union


class PerspectiveToEquirectPatch(nn.Module):
    """
    Lightweight perspective->equirectangular patch warp.

    - Samples a horizontal FOV (deg) from a range (approximate; avoids per-image FOV estimation).
    - Samples a square equirectangular window by choosing yaw/pitch center and yaw span; pitch span is half yaw span (to respect 2:1 equirect AR globally while output is square).
    - Applies a random 3D rotation (yaw, pitch, roll) and inverse-projects into the source perspective image.
    - Warps image with bilinear sampling and masks with nearest sampling.

    Output size is kept square to fit downstream pipelines; the equirect window spans are chosen accordingly.
    """

    def __init__(
        self,
        out_size: tuple[int, int],
        fov_deg_range: tuple[float, float] = (55.0, 90.0),
        yaw_span_deg_range: tuple[float, float] = (90.0, 180.0),
        rand_roll_deg: float = 10.0,
        prob: float = 1.0,
        quant_deg: float = 5.0,
        max_cache: int = 256,
    ) -> None:
        super().__init__()
        self.out_h, self.out_w = int(out_size[0]), int(out_size[1])
        self.fov_min, self.fov_max = float(fov_deg_range[0]), float(fov_deg_range[1])
        self.span_min, self.span_max = float(yaw_span_deg_range[0]), float(yaw_span_deg_range[1])
        self.rand_roll_deg = float(rand_roll_deg)
        self.prob = float(prob)
        self.quant_deg = float(quant_deg)
        self._grid_cache: dict[tuple, torch.Tensor] = {}
        self._max_cache = int(max_cache)

    def _sample_params(self) -> dict:
        fov_h = float(torch.empty(1).uniform_(self.fov_min, self.fov_max))
        yaw_span = float(torch.empty(1).uniform_(self.span_min, self.span_max))
        pitch_span = 0.5 * yaw_span

        # Center of the equirect window
        yaw0 = float(torch.empty(1).uniform_(-180.0, 180.0))
        pitch0 = float(torch.empty(1).uniform_(-60.0, 60.0))  # keep content on the sphere
        roll = float(torch.empty(1).uniform_(-self.rand_roll_deg, self.rand_roll_deg))

        return {
            "fov_h": fov_h,
            "yaw_span": yaw_span,
            "pitch_span": pitch_span,
            "yaw0": yaw0,
            "pitch0": pitch0,
            "roll": roll,
        }

    @staticmethod
    def _deg2rad(x: float) -> float:
        return x * math.pi / 180.0

    @staticmethod
    def _build_rotation(yaw_deg: float, pitch_deg: float, roll_deg: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        yaw = PerspectiveToEquirectPatch._deg2rad(yaw_deg)
        pitch = PerspectiveToEquirectPatch._deg2rad(pitch_deg)
        roll = PerspectiveToEquirectPatch._deg2rad(roll_deg)

        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        # R = Rz(roll) * Rx(pitch) * Ry(yaw)
        Ry = torch.tensor([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], device=device, dtype=dtype)
        Rx = torch.tensor([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], device=device, dtype=dtype)
        Rz = torch.tensor([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], device=device, dtype=dtype)
        return Rz @ (Rx @ Ry)

    def _make_grid(
        self,
        H_out: int,
        W_out: int,
        H_in: int,
        W_in: int,
        fov_h_deg: float,
        yaw_span_deg: float,
        pitch_span_deg: float,
        yaw0_deg: float,
        pitch0_deg: float,
        roll_deg: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Output pixel centers -> (u,v) in [-0.5,0.5]
        ys = torch.linspace(-0.5, 0.5, H_out, device=device, dtype=dtype)
        xs = torch.linspace(-0.5, 0.5, W_out, device=device, dtype=dtype)
        v, u = torch.meshgrid(ys, xs, indexing="ij")  # HxW

        # Window in equirect: yaw spans 2π, height π. For a square window:
        yaw_span = self._deg2rad(yaw_span_deg)
        pitch_span = self._deg2rad(pitch_span_deg)
        # Window centered at 0; camera orientation applied by rotation R below
        yaw = u * yaw_span
        pitch = v * pitch_span

        # Spherical -> 3D ray (Y up)
        cos_p = torch.cos(pitch)
        sin_p = torch.sin(pitch)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        X = cos_p * cos_y
        Y = sin_p
        Z = cos_p * sin_y
        dirs = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # (H*W, 3)

        # Apply rotation
        R = self._build_rotation(yaw0_deg, pitch0_deg, roll_deg, device=device, dtype=dtype)
        dirs = (dirs @ R.T)

        Xr = dirs[:, 0]
        Yr = -dirs[:, 1]  # flip Y to match image coords (down positive)
        Zr = dirs[:, 2]

        # Perspective projection parameters
        fov_h = self._deg2rad(fov_h_deg)
        fx = (W_in - 1) * 0.5 / math.tan(fov_h * 0.5)
        fy = fx * (H_in - 1) / max(1.0, (W_in - 1))  # aspect-consistent
        cx = (W_in - 1) * 0.5
        cy = (H_in - 1) * 0.5

        # Avoid divide-by-zero
        eps = 1e-6
        Zr_safe = torch.where(Zr.abs() < eps, torch.full_like(Zr, eps), Zr)

        x_img = fx * (Xr / Zr_safe) + cx
        y_img = fy * (Yr / Zr_safe) + cy

        # Normalize to [-1,1] for grid_sample
        x_norm = (x_img / max(1.0, (W_in - 1))) * 2.0 - 1.0
        y_norm = (y_img / max(1.0, (H_in - 1))) * 2.0 - 1.0

        # If point is behind the camera (Z<=0), push out of bounds so it pads to zeros
        behind = Zr <= 0.0
        x_norm = torch.where(behind, torch.full_like(x_norm, 2.0), x_norm)
        y_norm = torch.where(behind, torch.full_like(y_norm, 2.0), y_norm)

        grid = torch.stack([x_norm, y_norm], dim=-1).reshape(H_out, W_out, 2)
        return grid

    def _quant(self, x: float) -> float:
        q = self.quant_deg
        if q <= 0:
            return float(x)
        return round(x / q) * q

    def _get_grid_cached(
        self,
        H_out: int,
        W_out: int,
        H_in: int,
        W_in: int,
        fov_h_deg: float,
        yaw_span_deg: float,
        pitch_span_deg: float,
        yaw0_deg: float,
        pitch0_deg: float,
        roll_deg: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        # Quantize degrees to enable cache reuse
        fov_q = self._quant(fov_h_deg)
        yawspan_q = self._quant(yaw_span_deg)
        pitchspan_q = self._quant(pitch_span_deg)
        yaw0_q = self._quant(yaw0_deg)
        pitch0_q = self._quant(pitch0_deg)
        roll_q = self._quant(roll_deg)

        key = (
            H_out, W_out, H_in, W_in,
            fov_q, yawspan_q, pitchspan_q, yaw0_q, pitch0_q, roll_q,
        )
        grid = self._grid_cache.get(key)
        if grid is None:
            grid = self._make_grid(
                H_out, W_out, H_in, W_in,
                fov_q, yawspan_q, pitchspan_q, yaw0_q, pitch0_q, roll_q,
                device=torch.device("cpu"),
                dtype=torch.float32,
            ).cpu()
            # Evict if too large
            if len(self._grid_cache) >= self._max_cache:
                self._grid_cache.pop(next(iter(self._grid_cache)))
            self._grid_cache[key] = grid

        return grid.to(device=device, dtype=dtype)

    def forward(self, img: Tensor, target: dict[str, Any]):
        if self.prob < 1.0 and float(torch.rand(())) > self.prob:
            return img, target

        # Determine input image size and layout
        # Support CHW or HWC
        img_tensor = img
        orig_like = img
        img_dtype = torch.float32
        device = img_tensor.device

        if img_tensor.ndim == 3 and img_tensor.shape[0] in (1, 3):
            # CHW
            chw = True
            C, H_in, W_in = img_tensor.shape
            img_chw = img_tensor.to(dtype=img_dtype)
        elif img_tensor.ndim == 3 and img_tensor.shape[-1] in (1, 3):
            # HWC
            chw = False
            H_in, W_in, C = img_tensor.shape
            img_chw = img_tensor.permute(2, 0, 1).to(dtype=img_dtype)
        else:
            # Fallback: assume CHW
            chw = True
            C, H_in, W_in = img_tensor.shape[-3:]
            img_chw = img_tensor.to(dtype=img_dtype)

        params = self._sample_params()
        grid = self._get_grid_cached(
            self.out_h,
            self.out_w,
            H_in,
            W_in,
            params["fov_h"],
            params["yaw_span"],
            params["pitch_span"],
            params["yaw0"],
            params["pitch0"],
            params["roll"],
            device=device,
            dtype=img_chw.dtype,
        )

        grid_b = grid.unsqueeze(0)  # 1 x H_out x W_out x 2
        img_in = img_chw.unsqueeze(0)  # 1 x C x H_in x W_in

        warped_img = Fnn.grid_sample(
            img_in,
            grid_b,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        ).squeeze(0)

        # Masks: [B, H, W] bool -> as channels
        if "masks" in target and target["masks"] is not None and target["masks"].numel() > 0:
            masks = target["masks"]
            Bm, Hm, Wm = masks.shape
            masks_in = masks.to(dtype=img_dtype).unsqueeze(0)  # 1 x Bm x H x W
            warped_masks = Fnn.grid_sample(
                masks_in,
                grid_b,
                mode="nearest",
                padding_mode="zeros",
                align_corners=True,
            ).squeeze(0)  # Bm x H_out x W_out
            warped_masks = (warped_masks > 0.5)
            target = {
                **target,
                "masks": wrap(warped_masks, like=target["masks"]),
            }

        # Restore layout
        if chw:
            out_img = warped_img
        else:
            out_img = warped_img.permute(1, 2, 0)

        # Cast back to original dtype
        orig_dtype = img_tensor.dtype
        if orig_dtype == torch.uint8:
            out_img = out_img.clamp(0, 255).round().to(torch.uint8)
        else:
            out_img = out_img.to(orig_dtype)

        out_img = wrap(out_img, like=orig_like)
        return out_img, target


class Transforms(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        color_jitter_enabled: bool,
        scale_range: tuple[float, float],
        max_brightness_delta: int = 32,
        max_contrast_factor: float = 0.5,
        saturation_factor: float = 0.5,
        max_hue_delta: int = 18,
        enable_equirect_aug: bool = True,
        fov_deg_range: tuple[float, float] = (55.0, 90.0),
        yaw_span_deg_range: tuple[float, float] = (90.0, 150.0),
        rand_roll_deg: float = 10.0,
    ):
        super().__init__()

        self.img_size = img_size
        self.color_jitter_enabled = color_jitter_enabled
        self.max_brightness_factor = max_brightness_delta / 255.0
        self.max_contrast_factor = max_contrast_factor
        self.max_saturation_factor = saturation_factor
        self.max_hue_delta = max_hue_delta / 360.0

        self.random_horizontal_flip = T.RandomHorizontalFlip()
        self.scale_jitter = T.ScaleJitter(target_size=img_size, scale_range=scale_range)
        self.random_crop = T.RandomCrop(img_size)

        # Spherical augmentation: perspective -> equirectangular patch
        self.enable_equirect_aug = enable_equirect_aug
        if self.enable_equirect_aug:
            self.persp2equi = PerspectiveToEquirectPatch(
                out_size=img_size,
                fov_deg_range=fov_deg_range,
                yaw_span_deg_range=yaw_span_deg_range,
                rand_roll_deg=rand_roll_deg,
                prob=1.0,
            )

    def _random_factor(self, factor: float, center: float = 1.0):
        return torch.empty(1).uniform_(center - factor, center + factor).item()

    def _brightness(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_brightness(
                img, self._random_factor(self.max_brightness_factor)
            )

        return img

    def _contrast(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_contrast(img, self._random_factor(self.max_contrast_factor))

        return img

    def _saturation_and_hue(self, img):
        if torch.rand(()) < 0.5:
            img = F.adjust_saturation(
                img, self._random_factor(self.max_saturation_factor)
            )

        if torch.rand(()) < 0.5:
            img = F.adjust_hue(img, self._random_factor(self.max_hue_delta, center=0.0))

        return img

    def color_jitter(self, img):
        if not self.color_jitter_enabled:
            return img

        img = self._brightness(img)

        if torch.rand(()) < 0.5:
            img = self._contrast(img)
            img = self._saturation_and_hue(img)
        else:
            img = self._saturation_and_hue(img)
            img = self._contrast(img)

        return img

    def pad(
        self, img: Tensor, target: dict[str, Any]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        pad_h = max(0, self.img_size[-2] - img.shape[-2])
        pad_w = max(0, self.img_size[-1] - img.shape[-1])
        padding = [0, 0, pad_w, pad_h]

        img = F.pad(img, padding)
        target["masks"] = F.pad(target["masks"], padding)

        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        return {k: wrap(v[keep], like=v) for k, v in target.items()}

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]]
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img_orig, target_orig = img, target

        target = self._filter(target, ~target["is_crowd"])

        img = self.color_jitter(img)
        if getattr(self, "enable_equirect_aug", False):
            img, target = self.persp2equi(img, target)
        img, target = self.random_horizontal_flip(img, target)
        img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        img, target = self.random_crop(img, target)

        valid = target["masks"].flatten(1).any(1)
        if not valid.any():
            return self(img_orig, target_orig)

        target = self._filter(target, valid)

        return img, target

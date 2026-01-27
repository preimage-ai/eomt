# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
#
# Portions of this file are adapted from Detectron2 by Facebook, Inc. and its affiliates,
# used under the Apache 2.0 License.
# ---------------------------------------------------------------

import torch
import cv2
import numpy as np
from torchvision.transforms import v2 as T
from torchvision.transforms.v2 import functional as F
from torchvision.tv_tensors import wrap, TVTensor, Image as TVImage, Mask as TVMask
from torch import nn, Tensor
from typing import Any, Union, Dict

# Import equilib for perspective-to-equirectangular transformation
import sys
import os
equilib_path = os.path.join(os.path.dirname(__file__), '..', 'equilib')
sys.path.append(equilib_path)
from equilib import Pers2Equi


def crop_black_regions_with_coords(img: Tensor) -> tuple[Tensor, tuple]:
    """
    Automatically crop black regions from the edges of an image and return crop coordinates.
    
    Args:
        img: Input image tensor (C, H, W) or (H, W, C)
        
    Returns:
        Tuple of (cropped_image_tensor, crop_coordinates)
        crop_coordinates: (y_min, y_max, x_min, x_max) or None if no cropping
    """
    # Convert to numpy for easier processing
    if img.dim() == 3:
        if img.shape[0] <= 4:  # Likely (C, H, W) format
            img_np = img.permute(1, 2, 0).cpu().numpy()
            is_chw = True
        elif img.shape[2] <= 4:  # Likely (H, W, C) format
            img_np = img.cpu().numpy()
            is_chw = False
        else:
            return img, None  # Unusual format, return original
    else:
        return img, None  # Can't process other dimensions
    
    # Handle different data types
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    
    # Convert to grayscale for edge detection
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    elif len(img_np.shape) == 3 and img_np.shape[2] == 1:
        gray = img_np.squeeze(2)
    elif len(img_np.shape) == 2:
        gray = img_np
    else:
        return img, None  # Unsupported format, return original
    
    # Find non-black regions
    mask = gray > 10  # Threshold for black pixels
    
    if not mask.any():
        return img, None  # No non-black pixels found, return original
    
    # Find bounding box of non-black regions
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not rows.any() or not cols.any():
        return img, None  # No valid content found
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # Add small padding to avoid cutting off content
    padding = 5
    y_min = max(0, y_min - padding)
    y_max = min(gray.shape[0], y_max + padding)
    x_min = max(0, x_min - padding)
    x_max = min(gray.shape[1], x_max + padding)
    
    # Crop the image
    cropped = img_np[y_min:y_max, x_min:x_max]
    
    # Convert back to tensor with original format
    if is_chw:
        cropped_tensor = torch.from_numpy(cropped).permute(2, 0, 1).to(img.device)
    else:
        cropped_tensor = torch.from_numpy(cropped).to(img.device)
    
    # Convert back to original dtype
    if img.dtype == torch.uint8:
        cropped_tensor = cropped_tensor.to(torch.uint8)
    else:
        cropped_tensor = cropped_tensor.float()
        if img.max() <= 1.0:
            cropped_tensor = cropped_tensor / 255.0
    
    return cropped_tensor, (y_min, y_max, x_min, x_max)


def apply_crop_coords(mask: Tensor, crop_coords: tuple) -> Tensor:
    """
    Apply the same crop coordinates to a mask tensor.
    
    Args:
        mask: Input mask tensor
        crop_coords: (y_min, y_max, x_min, x_max) coordinates
        
    Returns:
        Cropped mask tensor
    """
    y_min, y_max, x_min, x_max = crop_coords
    
    # Convert to numpy for cropping
    if mask.dim() == 2:
        mask_np = mask.cpu().numpy()
        cropped_np = mask_np[y_min:y_max, x_min:x_max]
        cropped_tensor = torch.from_numpy(cropped_np).to(mask.device)
    elif mask.dim() == 3:
        if mask.shape[0] <= 4:  # (C, H, W) format
            mask_np = mask.permute(1, 2, 0).cpu().numpy()
            cropped_np = mask_np[y_min:y_max, x_min:x_max]
            cropped_tensor = torch.from_numpy(cropped_np).permute(2, 0, 1).to(mask.device)
        else:  # (H, W, C) format
            mask_np = mask.cpu().numpy()
            cropped_np = mask_np[y_min:y_max, x_min:x_max]
            cropped_tensor = torch.from_numpy(cropped_np).to(mask.device)
    else:
        return mask  # Unsupported format, return original
    
    return cropped_tensor.to(mask.dtype)


def fast_blur_padding(img: Tensor, blur_kernel_size: int = 5, blur_sigma: float = 1.0) -> Tensor:
    """
    Fast blur padding to replace black pixels with blurred content.
    Uses efficient Gaussian blur and inpainting-like approach for speed.
    
    Args:
        img: Input image tensor (C, H, W) or (H, W, C)
        blur_kernel_size: Size of Gaussian blur kernel
        blur_sigma: Standard deviation for Gaussian blur
        
    Returns:
        Image tensor with blur padding instead of black pixels
    """
    # Convert to numpy for OpenCV operations
    if img.dim() == 3:
        if img.shape[0] <= 4:  # Likely (C, H, W) format
            img_np = img.permute(1, 2, 0).cpu().numpy()
            is_chw = True
        elif img.shape[2] <= 4:  # Likely (H, W, C) format
            img_np = img.cpu().numpy()
            is_chw = False
        else:
            return img  # Unusual format, return original
    else:
        return img  # Can't process other dimensions
    
    # Handle different data types
    original_dtype = img_np.dtype
    if img_np.dtype != np.uint8:
        if img_np.max() <= 1.0:
            img_np = (img_np * 255).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)
    
    # Create mask of black/invalid pixels (near black)
    if len(img_np.shape) == 3:
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_np
    
    # Black pixel mask (threshold for near-black pixels)
    black_mask = gray < 20
    
    if not black_mask.any():
        # No black pixels to fill, return original
        result_tensor = torch.from_numpy(img_np).to(img.device)
        if is_chw:
            result_tensor = result_tensor.permute(2, 0, 1)
        return result_tensor.to(img.dtype)
    
    # Fast approach: use dilated border pixels and blur
    # 1. Create a mask of valid (non-black) regions
    valid_mask = ~black_mask
    
    # 2. Dilate the valid mask to expand borders slightly
    kernel = np.ones((3, 3), np.uint8)
    dilated_mask = cv2.dilate(valid_mask.astype(np.uint8), kernel, iterations=2).astype(bool)
    
    # 3. Create a blurred version of the entire image
    blurred_img = cv2.GaussianBlur(img_np, (blur_kernel_size, blur_kernel_size), blur_sigma)
    
    # 4. Use the blurred image only where we need to fill black regions
    # but only from areas that were originally valid (dilated mask)
    result = img_np.copy()
    fill_regions = black_mask & dilated_mask
    result[fill_regions] = blurred_img[fill_regions]
    
    # 5. For any remaining black pixels (corners, etc.), use a simple edge extension
    if black_mask.any():
        # Use inpainting for remaining holes (fast OpenCV implementation)
        inpaint_mask = black_mask.astype(np.uint8) * 255
        if len(img_np.shape) == 3:
            result = cv2.inpaint(result, inpaint_mask, 3, cv2.INPAINT_TELEA)
        else:
            result = cv2.inpaint(result, inpaint_mask, 3, cv2.INPAINT_TELEA)
    
    # Convert back to tensor with original format
    if is_chw:
        result_tensor = torch.from_numpy(result).permute(2, 0, 1).to(img.device)
    else:
        result_tensor = torch.from_numpy(result).to(img.device)
    
    # Convert back to original dtype
    if img.dtype == torch.uint8:
        result_tensor = result_tensor.to(torch.uint8)
    else:
        result_tensor = result_tensor.float()
        if img.max() <= 1.0:
            result_tensor = result_tensor / 255.0
    
    return result_tensor


class PanoramicDistortion(nn.Module):
    """
    Panoramic distortion augmentation using perspective-to-equirectangular transformation.
    Applies subtle panoramic distortion to simulate camera movement and perspective changes.
    """
    
    def __init__(
        self,
        intensity: float = 0.3,
        apply_prob: float = 0.5,
        fov_deg: float = 90.0,
        vertical_intensity: float = None,
        horizontal_intensity: float = None,
        output_size: tuple = None,  # Will be calculated dynamically
    ):
        super().__init__()
        self.intensity = intensity
        self.apply_prob = apply_prob
        self.fov_deg = fov_deg
        
        # Use separate intensities if provided, otherwise use base intensity
        self.vertical_intensity = vertical_intensity if vertical_intensity is not None else intensity
        self.horizontal_intensity = horizontal_intensity if horizontal_intensity is not None else intensity
        
        # Output size will be calculated dynamically based on input aspect ratio
        self.output_size = output_size
        
        # Initialize pers2equi converter - will be created dynamically
        self.pers2equi = None

    def forward(
        self,
        img: Tensor,
        target: Dict[str, Union[Tensor, TVTensor]],
        img_path: str = None,  # Optional image path for filename checking
    ) -> tuple[Tensor, Dict[str, Union[Tensor, TVTensor]]]:

        if torch.rand(()) > self.apply_prob:
            return img, target
        
        # Skip panoramic distortion if "equi" is in the image path
        if img_path and "equi" in img_path.lower():
            return img, target

        h, w = img.shape[-2:]
        device = img.device
        
        # Calculate output size for equirectangular projection
        # Input is a cubemap face (n x n), output should be equirectangular dimensions
        if self.output_size is None:
            # For equirectangular: height = input_height, width = 2 * input_height
            # This gives proper equirectangular aspect ratio (2:1)
            output_height = h  # Same height as input face
            output_width = 2 * h  # Double width for equirectangular
        else:
            output_height, output_width = self.output_size
        
        # Initialize pers2equi converter if not already done or if size changed
        if (self.pers2equi is None or 
            self.pers2equi.height != output_height or 
            self.pers2equi.width != output_width):
            self.pers2equi = Pers2Equi(
                height=output_height,
                width=output_width,
                mode="bilinear",
                clip_output=True,
            )
        
        # Store original dtype
        orig_dtype = img.dtype
        
        # Convert to float if needed
        img_float = img.float() if orig_dtype == torch.uint8 else img
        
        # Create rotation using random values multiplied by intensity
        # This simulates the distortion by rotating the virtual camera
        yaw = torch.randn(()) * self.horizontal_intensity  # Random yaw
        pitch = torch.randn(()) * self.vertical_intensity   # Random pitch
        roll = 0.0  # No roll for clean distortion
        
        # Create random FOV with clipping to range [75, 120]
        random_fov = torch.randn(()) * 15.0 + self.fov_deg  # Random variation around base FOV
        random_fov = torch.clamp(random_fov, 87.0, 92.0)   # Clip to valid range
        
        # Create rotation dictionary
        rots = [{"yaw": yaw, "pitch": pitch, "roll": roll}]
        
        try:
            # Apply perspective-to-equirectangular transformation
            # Add batch dimension if needed
            if img_float.dim() == 3:
                img_batch = img_float.unsqueeze(0)
            else:
                img_batch = img_float
            
            equi_img = self.pers2equi(
                pers=img_batch,
                rots=rots,
                fov_x=random_fov,  # Use random FOV
            )
            
            # Remove batch dimension if it was added
            if img_float.dim() == 3:
                equi_img = equi_img.squeeze(0)
            
            # Apply cropping to minimize blur area, then blur padding for remaining edges
            equi_img, crop_coords = crop_black_regions_with_coords(equi_img)
            
            # Apply minimal blur padding only to any remaining edge artifacts
            equi_img = fast_blur_padding(equi_img, blur_kernel_size=3, blur_sigma=0.5)
            
            # Resize back to original dimensions to maintain size consistency
            if equi_img.shape[-2:] != (h, w):
                equi_img = F.resize(equi_img, (h, w), antialias=True)
            
            # Convert back to original dtype
            if orig_dtype == torch.uint8:
                equi_img = (equi_img * 255.0).clamp(0, 255).to(torch.uint8)
            
            # Apply to masks if present
            if "masks" in target:
                masks = target["masks"]
                
                # Handle different mask formats
                if masks.dim() == 2:
                    # Single mask (H, W) - add batch and channel dims
                    masks_batch = masks.unsqueeze(0).unsqueeze(0)
                elif masks.dim() == 3:
                    if masks.shape[0] == 1:
                        # Single mask with channel dim (1, H, W) - add batch dim
                        masks_batch = masks.unsqueeze(0)
                    else:
                        # Multiple masks (N, H, W) - add channel dim
                        masks_batch = masks.unsqueeze(1)
                elif masks.dim() == 4:
                    # Already in batch format (B, C, H, W)
                    masks_batch = masks
                else:
                    raise ValueError(f"Unsupported mask tensor dimensions: {masks.dim()}")
                
                # Process each mask instance separately to avoid rotation mismatch
                processed_masks = []
                for i in range(masks_batch.shape[0]):
                    single_mask = masks_batch[i:i+1]  # Keep batch dim
                    
                    # Convert to float for pers2equi
                    masks_float = single_mask.float()
                    
                    # Use nearest-neighbor interpolation for masks to keep them binary
                    # We need to create a separate Pers2Equi instance for masks with nearest mode
                    pano_masks = Pers2Equi(
                        height=output_height,
                        width=output_width,
                        mode="nearest",  # Use nearest-neighbor for masks
                        clip_output=True,
                    )
                    
                    equi_masks = pano_masks(
                        pers=masks_float,
                        rots=rots,
                        fov_x=random_fov,  # Use same random FOV
                    )
                    
                    # Remove batch dim
                    if masks.dim() == 2:
                        equi_mask = equi_masks.squeeze(0).squeeze(0)
                    elif masks.dim() == 3:
                        if masks.shape[0] == 1:
                            equi_mask = equi_masks.squeeze(0).squeeze(0)
                        else:
                            equi_mask = equi_masks.squeeze(0).squeeze(1)
                    elif masks.dim() == 4:
                        equi_mask = equi_masks.squeeze(0).squeeze(1)
                    
                    # Apply the SAME crop coordinates to ensure 1-to-1 correspondence
                    if crop_coords is not None:
                        equi_mask = apply_crop_coords(equi_mask, crop_coords)
                    
                    # For masks, set any remaining black regions to background (0)
                    # since blur padding doesn't make sense for binary masks
                    mask_gray = equi_mask.squeeze() if equi_mask.dim() > 2 else equi_mask
                    if mask_gray.dim() == 2:
                        # Create mask of black regions and set to background
                        black_mask = mask_gray < 0.5  # Threshold for background
                        if black_mask.any():
                            mask_gray[black_mask] = 0.0
                        equi_mask = mask_gray
                    
                    # Resize mask back to original dimensions
                    if equi_mask.shape[-2:] != (h, w):
                        equi_mask = F.resize(equi_mask.unsqueeze(0), (h, w), antialias=True).squeeze(0)
                    
                    # Ensure mask remains binary (0 or 1) by rounding
                    equi_mask = torch.round(equi_mask)
                    
                    processed_masks.append(equi_mask)
                
                # Stack all processed masks
                if masks.dim() == 2:
                    equi_masks = processed_masks[0]  # Single mask
                elif masks.dim() == 3:
                    if masks.shape[0] == 1:
                        equi_masks = processed_masks[0]  # Single mask
                    else:
                        equi_masks = torch.stack(processed_masks)  # Multiple masks
                elif masks.dim() == 4:
                    equi_masks = torch.stack(processed_masks)  # Multiple masks
                
                # Convert back to original mask dtype
                target["masks"] = equi_masks.to(masks.dtype)
            
            return equi_img, target
            
        except Exception as e:
            print(f"⚠️ Error in pers2equi transformation: {e}")
            # Fallback to original image if transformation fails
            return img, target


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
        # Panoramic distortion parameters
        panoramic_enabled: bool = False,
        panoramic_intensity: float = 0.3,
        panoramic_apply_prob: float = 0.1,
        panoramic_fov_deg: float = 90.0,
        panoramic_vertical_intensity: float = 0.15,
        panoramic_horizontal_intensity: float = 0.0,
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
        
        # Initialize panoramic distortion if enabled
        if panoramic_enabled:
            self.panoramic_distortion = PanoramicDistortion(
                intensity=panoramic_intensity,
                apply_prob=panoramic_apply_prob,
                fov_deg=panoramic_fov_deg,
                vertical_intensity=panoramic_vertical_intensity,
                horizontal_intensity=panoramic_horizontal_intensity,
            )
        else:
            self.panoramic_distortion = None

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
        
        # Handle masks with different shapes after panoramic distortion
        masks = target["masks"]
        if masks.dim() == 4 and masks.shape[1] == 1:
            # Mask has extra channel dimension (N, 1, H, W) - squeeze it for padding
            masks_squeezed = masks.squeeze(1)
            masks_padded = F.pad(masks_squeezed, padding)
            # Add channel dimension back
            target["masks"] = masks_padded.unsqueeze(1)
        elif masks.dim() == 2:
            # Handle 2D mask (H, W) - add batch and channel dimensions
            masks = masks.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
            masks_padded = F.pad(masks, padding)
            target["masks"] = masks_padded.squeeze(0)  # Back to (1, H, W)
        else:
            # Normal mask padding (should be 3D: 1, H, W)
            if masks.dim() == 3 and masks.shape[0] == 1:
                # Add batch dimension if missing
                masks = masks.unsqueeze(0)
            target["masks"] = F.pad(masks, padding)

        return img, target

    def _filter(self, target: dict[str, Union[Tensor, TVTensor]], keep: Tensor) -> dict:
        # Filter tensors, but preserve non-tensor metadata like is_rooftop
        filtered = {}
        for k, v in target.items():
            if isinstance(v, (Tensor, TVTensor)):
                filtered[k] = wrap(v[keep], like=v)
            else:
                # Preserve non-tensor values (e.g., is_rooftop boolean)
                filtered[k] = v
        return filtered

    def forward(
        self, img: Tensor, target: dict[str, Union[Tensor, TVTensor]], img_path: str = None
    ) -> tuple[Tensor, dict[str, Union[Tensor, TVTensor]]]:
        img_orig, target_orig = img, target

        target = self._filter(target, ~target["is_crowd"])

        # Apply panoramic distortion first (if enabled)
        if self.panoramic_distortion is not None:
            img, target = self.panoramic_distortion(img, target, img_path)

        img = self.color_jitter(img)
        img, target = self.random_horizontal_flip(img, target)
        img, target = self.scale_jitter(img, target)
        img, target = self.pad(img, target)
        img, target = self.random_crop(img, target)

        valid = target["masks"].flatten(1).any(1)
        if not valid.any():
            return self(img_orig, target_orig)

        target = self._filter(target, valid)

        return img, target

#!/usr/bin/env python3
"""
EOMT Segmentation Inference with Feature Map Visualization
"""

import os
import sys
import importlib
import argparse
import yaml
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from loguru import logger
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple

class EOMTSegmentationInference:
    def __init__(
        self, 
        config_path: str, 
        checkpoint_path: str, 
        device: str = "cuda:0",
        save_feature_maps: bool = False,
        feature_map_dir: Optional[str] = None,
        use_windowed: bool = False,
        window_size: int = 512
    ):
        """
        Initialize EOMT segmentation model for inference
        
        Args:
            config_path: Path to EOMT config YAML file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            save_feature_maps: Whether to save intermediate feature maps
            feature_map_dir: Directory to save feature maps
            use_windowed: Whether to use windowed inference for large images
            window_size: Size of sliding window (default: 512)
        """
        self.device = device
        self.img_size = (window_size, window_size) if use_windowed else (512, 1024)
        self.use_windowed = use_windowed
        self.window_size = window_size
        self.save_feature_maps = save_feature_maps
        self.feature_map_dir = Path(feature_map_dir) if feature_map_dir else None
        self.feature_maps = {}
        self.handles = []
        
        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        logger.info("Building EOMT model...")
        self._build_model_from_checkpoint(checkpoint_path)
        
        if self.save_feature_maps:
            self._register_feature_hooks()
            
        self.model.eval()
        logger.info(f"✓ Model initialized on {device}")

    def _fix_import_path(self, class_path: str) -> str:
        """Fix import paths from config"""
        replacements = {
            'models.vit.ViT': 'models.eomt.vit.ViT',
            'models.eomt.EoMT': 'models.eomt.eomt.EoMT',
        }
        return replacements.get(class_path, class_path)

    def _infer_num_classes_from_checkpoint(self, state_dict: dict) -> int:
        """Infer number of classes from checkpoint"""
        for key in state_dict.keys():
            if 'class_head.weight' in key:
                checkpoint_classes = state_dict[key].shape[0]
                # EoMT adds +1 internally
                actual_num_classes = checkpoint_classes - 1
                logger.info(f"Checkpoint: {checkpoint_classes} output classes, using num_classes={actual_num_classes}")
                return actual_num_classes
        
        logger.warning("Could not infer num_classes, using 150")
        return 150

    def _build_model_from_checkpoint(self, checkpoint_path: str):
        """Build model and load checkpoint"""
        # Load checkpoint first
        logger.info(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint:
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint
        
        # Remove DDP prefix
        torch.nn.modules.utils.consume_prefix_in_state_dict_if_present(state_dict, 'module.')
        
        # Infer num_classes
        num_classes = self._infer_num_classes_from_checkpoint(state_dict)
        
        # Import modules
        network_cfg = self.config["model"]["init_args"]["network"]
        network_path = self._fix_import_path(network_cfg["class_path"])
        network_module_name, network_class_name = network_path.rsplit(".", 1)
        
        network_module = importlib.import_module(network_module_name)
        EoMT = getattr(network_module, network_class_name)
        
        # Build encoder
        encoder_cfg = network_cfg["init_args"]["encoder"]
        encoder_path = self._fix_import_path(encoder_cfg["class_path"])
        encoder_module_name, encoder_class_name = encoder_path.rsplit(".", 1)
        
        encoder_module = importlib.import_module(encoder_module_name)
        ViT = getattr(encoder_module, encoder_class_name)
        
        encoder_init_args = encoder_cfg.get("init_args", {}).copy()
        
        logger.info("Building encoder...")
        encoder = ViT(img_size=self.img_size, **encoder_init_args)
        
        # Build EoMT network
        network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
        logger.info(f"Building EoMT with num_classes={num_classes}")

        # Create a simple wrapper
        class SimpleWrapper(torch.nn.Module):
            def __init__(self, eomt_network):
                super().__init__()
                self.network = eomt_network
            
            def forward(self, x):
                return self.network(x)
        
        eomt_network = EoMT(
            encoder=encoder,
            num_classes=num_classes,
            masked_attn_enabled=False,
            **network_kwargs
        )
        
        self.model = SimpleWrapper(eomt_network).to(self.device)
        
        # Load checkpoint weights
        logger.info("Loading weights from checkpoint...")
        
        # Fix state dict keys
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('network.'):
                fixed_state_dict[key] = value
            else:
                fixed_state_dict['network.' + key] = value
        
        # Load weights
        missing, unexpected = self.model.load_state_dict(fixed_state_dict, strict=False)
        logger.info(f"✓ Loaded checkpoint: {len(missing)} missing keys, {len(unexpected)} unexpected keys")
        
        if len(missing) > 300 or len(unexpected) > 300:
            logger.warning("Large number of mismatched keys - results may be poor")
            logger.warning(f"Sample missing: {missing[:5]}")
            logger.warning(f"Sample unexpected: {unexpected[:5]}")

    def _register_feature_hooks(self):
        """Register forward hooks to capture feature maps from model layers"""
        def get_activation(name):
            def hook(model, input, output):
                # Handle both tensor and tuple outputs
                if isinstance(output, torch.Tensor):
                    self.feature_maps[name] = output.detach().cpu()
                elif isinstance(output, tuple):
                    # If output is a tuple, save each tensor in the tuple
                    for i, out in enumerate(output):
                        if isinstance(out[0], torch.Tensor):
                            self.feature_maps[f"{name}.{i}"] = out[0].detach().cpu()
            return hook

        # Clear any existing hooks
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.feature_maps = {}

        # Register hooks for the network
        if hasattr(self.model.network, 'encoder'):
            # Register hooks for each encoder block
            for name, layer in self.model.network.encoder.named_children():
                if isinstance(layer, torch.nn.ModuleList):
                    for i, block in enumerate(layer):
                        self.handles.append(block.register_forward_hook(get_activation(f'encoder.{i}')))
                else:
                    self.handles.append(layer.register_forward_hook(get_activation(f'encoder.{name}')))

        # Register hooks for decoder if exists
        if hasattr(self.model.network, 'decoder'):
            for name, layer in self.model.network.decoder.named_children():
                self.handles.append(layer.register_forward_hook(get_activation(f'decoder.{name}')))

        # Also register hooks for the main network
        self.handles.append(self.model.network.register_forward_hook(get_activation('network')))

    def _save_feature_maps(self, image_path: str):
        """Save captured feature maps to disk"""
        if not self.feature_maps or not self.save_feature_maps:
            return

        # Create output directory
        if self.feature_map_dir is None:
            output_dir = Path(image_path).parent / 'feature_maps'
        else:
            output_dir = self.feature_map_dir / 'feature_maps'
        
        output_dir.mkdir(parents=True, exist_ok=True)
        base_name = Path(image_path).stem

        for layer_name, feat in self.feature_maps.items():
            # Skip if feature is not a 4D tensor [B, C, H, W]
            if len(feat.shape) != 4:
                continue
                
            # Take first batch element and average across channels
            feat = feat[0].mean(0).numpy()
            
            # Normalize for visualization
            feat = (feat - feat.min()) / (feat.max() - feat.min() + 1e-8)
            
            # Create plot
            plt.figure(figsize=(10, 10))
            plt.imshow(feat, cmap='viridis')
            plt.colorbar()
            plt.title(f'Feature Map: {layer_name}')
            plt.axis('off')
            
            # Save figure
            safe_layer_name = layer_name.replace('.', '_')
            output_path = output_dir / f'{base_name}_{safe_layer_name}.png'
            plt.savefig(str(output_path), bbox_inches='tight', dpi=100)
            plt.close()

    def preprocess_image(self, image_path: str) -> Tuple[torch.Tensor, np.ndarray, tuple]:
        """Load and preprocess image"""
        img = Image.open(image_path).convert('RGB')
        orig_img = np.array(img)
        orig_size = (orig_img.shape[0], orig_img.shape[1])
        
        if self.use_windowed:
            # For windowed inference, keep original size as tensor
            img_tensor = torch.from_numpy(orig_img).float()
            img_tensor = img_tensor.permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
        else:
            # Direct resize for non-windowed inference
            img_resized = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
            img_tensor = torch.from_numpy(np.array(img_resized)).float()
            img_tensor = img_tensor.permute(2, 0, 1) / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, orig_img, orig_size

    def _window_image(self, img_tensor: torch.Tensor) -> Tuple[List[torch.Tensor], List[Tuple[int, int, int]]]:
        """Split image into overlapping windows for inference"""
        crops, origins = [], []
        
        # img_tensor shape: [1, C, H, W]
        img = img_tensor[0]  # [C, H, W]
        _, h, w = img.shape
        
        # Calculate number of crops needed
        num_crops_h = math.ceil(h / self.window_size)
        num_crops_w = math.ceil(w / self.window_size)
        
        # Calculate overlap to cover entire image
        overlap_h = (num_crops_h * self.window_size - h) / max(num_crops_h - 1, 1) if num_crops_h > 1 else 0
        overlap_w = (num_crops_w * self.window_size - w) / max(num_crops_w - 1, 1) if num_crops_w > 1 else 0
        
        for i in range(num_crops_h):
            for j in range(num_crops_w):
                # Calculate crop position with overlap
                start_h = int(i * (self.window_size - overlap_h))
                start_w = int(j * (self.window_size - overlap_w))
                end_h = min(start_h + self.window_size, h)
                end_w = min(start_w + self.window_size, w)
                
                # Adjust start if we're at the edge
                if end_h == h:
                    start_h = max(0, h - self.window_size)
                if end_w == w:
                    start_w = max(0, w - self.window_size)
                
                # Extract crop
                crop = img[:, start_h:end_h, start_w:end_w]
                
                # Pad if necessary (edge cases)
                if crop.shape[1] < self.window_size or crop.shape[2] < self.window_size:
                    pad_h = self.window_size - crop.shape[1]
                    pad_w = self.window_size - crop.shape[2]
                    crop = F.pad(crop, (0, pad_w, 0, pad_h), mode='reflect')
                
                crops.append(crop.unsqueeze(0))  # Add batch dimension
                origins.append((start_h, end_h, start_w, end_w))
        
        return crops, origins
    
    def _stitch_windows(self, predictions: List[torch.Tensor], origins: List[Tuple[int, int, int, int]], 
                       orig_size: Tuple[int, int]) -> torch.Tensor:
        """Stitch windowed predictions back into full image"""
        h, w = orig_size
        num_classes = predictions[0].shape[0]
        
        # Create output tensor and count tensor for averaging overlaps
        output = torch.zeros((num_classes, h, w), device=predictions[0].device)
        counts = torch.zeros((h, w), device=predictions[0].device)
        
        for pred, (start_h, end_h, start_w, end_w) in zip(predictions, origins):
            # Handle potential padding
            pred_h = end_h - start_h
            pred_w = end_w - start_w
            
            # Add prediction to output (averaging overlaps)
            output[:, start_h:end_h, start_w:end_w] += pred[:, :pred_h, :pred_w]
            counts[start_h:end_h, start_w:end_w] += 1
        
        # Average overlapping regions
        output = output / counts.unsqueeze(0).clamp(min=1)
        
        return output

    @torch.no_grad()
    def predict(self, image_path: str, save_feature_maps: Optional[bool] = None) -> np.ndarray:
        """Run segmentation inference with optional feature map saving"""
        if save_feature_maps is not None:
            self.save_feature_maps = save_feature_maps
        
        # Clear previous feature maps and re-register hooks
        self.feature_maps.clear()
        if self.save_feature_maps:
            self._register_feature_hooks()
        
        img_tensor, orig_img, orig_size = self.preprocess_image(image_path)
        
        with torch.inference_mode(), torch.amp.autocast(device_type='cuda' if 'cuda' in self.device else 'cpu', 
                                                    enabled=True, dtype=torch.float16):
            if self.use_windowed:
                # Windowed inference for large images
                crops, origins = self._window_image(img_tensor)
                
                all_predictions = []
                for crop in crops:
                    # Forward pass on each crop
                    mask_logits_per_layer, class_logits_per_layer = self.model.network(crop)
                    
                    # Get prediction for this crop
                    mask_logits = F.interpolate(
                        mask_logits_per_layer[-1], (self.window_size, self.window_size), mode="bilinear"
                    )
                    
                    B, Q, H, W = mask_logits.shape
                    mask_probs = mask_logits.sigmoid()
                    class_probs = class_logits_per_layer[-1].softmax(dim=-1)[..., :-1]  # Exclude void
                    per_pixel_logits = torch.einsum('bqhw,bqc->bchw', mask_probs, class_probs)
                    
                    all_predictions.append(per_pixel_logits[0])  # Remove batch dimension
                
                # Stitch predictions together
                stitched_logits = self._stitch_windows(all_predictions, origins, orig_size)
                preds = stitched_logits.argmax(0).cpu().numpy()
                
                # Save feature maps if enabled (only from last crop)
                if self.save_feature_maps and self.feature_maps:
                    self._save_feature_maps(image_path)
                    logger.info(f"Saved {len(self.feature_maps)} feature maps to {self.feature_map_dir or 'feature_maps'}")
            else:
                # Direct inference (original behavior)
                mask_logits_per_layer, class_logits_per_layer = self.model.network(img_tensor)
                
                # Save feature maps if enabled
                if self.save_feature_maps and self.feature_maps:
                    self._save_feature_maps(image_path)
                    logger.info(f"Saved {len(self.feature_maps)} feature maps to {self.feature_map_dir or 'feature_maps'}")
                
                # Rest of the prediction logic...
                mask_logits = F.interpolate(
                    mask_logits_per_layer[-1], self.img_size, mode="bilinear"
                )
                
                B, Q, H, W = mask_logits.shape
                mask_probs = mask_logits.sigmoid()
                class_probs = class_logits_per_layer[-1].softmax(dim=-1)[..., :-1]  # Exclude void
                per_pixel_logits = torch.einsum('bqhw,bqc->bchw', mask_probs, class_probs)
                
                preds = per_pixel_logits.argmax(1)[0].cpu().numpy()
                
                # Resize to original size
                if preds.shape != orig_size:
                    preds = cv2.resize(
                        preds.astype(np.uint8),
                        (orig_size[1], orig_size[0]),
                        interpolation=cv2.INTER_NEAREST
                    )
        
        return preds

    def save_segmentation(self, seg_mask: np.ndarray, output_path: str, colormap: bool = True):
        """Save segmentation mask"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if colormap:
            seg_colored = self.apply_colormap(seg_mask)
            cv2.imwrite(str(output_path), cv2.cvtColor(seg_colored, cv2.COLOR_RGB2BGR))
        else:
            np.save(str(output_path.with_suffix('.npy')), seg_mask)
            cv2.imwrite(str(output_path), seg_mask.astype(np.uint8))

    @staticmethod
    def apply_colormap(seg_mask: np.ndarray, num_classes: int = 167) -> np.ndarray:
        """Apply colormap to segmentation mask"""
        np.random.seed(42)
        colormap = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
        colormap[0] = [0, 0, 0]  # Set background to black
        
        h, w = seg_mask.shape
        seg_colored = np.zeros((h, w, 3), dtype=np.uint8)
        max_class = int(seg_mask.max())
        for class_idx in range(min(num_classes, max_class + 1)):
            seg_colored[seg_mask == class_idx] = colormap[class_idx]
        
        return seg_colored

    def __del__(self):
        """Clean up hooks when the object is destroyed"""
        for handle in self.handles:
            handle.remove()

def main():
    parser = argparse.ArgumentParser(description="EOMT Segmentation Inference with Feature Map Visualization")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing input images")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Directory to save output segmentation masks")
    parser.add_argument("--config", type=str, default="configs/eomt_large_512.yaml",
                       help="Path to model config file")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run inference on (e.g., 'cuda:0' or 'cpu')")
    parser.add_argument("--colormap", action="store_true",
                       help="Save colorized segmentation masks")
    parser.add_argument("--save_raw", action="store_true",
                       help="Save raw segmentation masks")
    parser.add_argument("--save_feature_maps", action="store_true",
                       help="Save intermediate feature maps")
    parser.add_argument("--feature_map_dir", type=str, default=None,
                       help="Directory to save feature maps (default: input_dir/feature_maps)")
    parser.add_argument("--use_windowed", action="store_true",
                       help="Use windowed inference for large images (recommended for ERP)")
    parser.add_argument("--window_size", type=int, default=512,
                       help="Window size for windowed inference (default: 512)")
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"],
                       help="Image file extensions to process")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    # Create output subdirectories
    if args.colormap:
        (output_dir / "colored").mkdir(exist_ok=True)
    if args.save_raw:
        (output_dir / "raw").mkdir(exist_ok=True)
    
    try:
        # Initialize model
        model = EOMTSegmentationInference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device,
            save_feature_maps=args.save_feature_maps,
            feature_map_dir=args.feature_map_dir,
            use_windowed=args.use_windowed,
            window_size=args.window_size
        )
        
        if args.use_windowed:
            logger.info(f"Using windowed inference with window size: {args.window_size}x{args.window_size}")
        else:
            logger.info("Using direct inference (resizing to model input size)")
        
        # Find all image files
        image_paths = []
        for ext in args.extensions:
            image_paths.extend(list(input_dir.glob(f"*{ext}")))
        
        if not image_paths:
            logger.error(f"No images found with extensions {args.extensions} in {input_dir}")
            sys.exit(1)
        
        logger.info(f"Found {len(image_paths)} images to process")
        
        # Process each image
        for img_path in tqdm(image_paths, desc="Processing images"):
            try:
                # Run inference
                pred = model.predict(str(img_path))
                
                # Save results
                rel_path = img_path.relative_to(input_dir)
                if args.colormap:
                    output_path = output_dir / "colored" / rel_path.with_suffix('.png')
                    model.save_segmentation(pred, str(output_path), colormap=True)
                
                if args.save_raw:
                    output_path = output_dir / "raw" / rel_path.with_suffix('.png')
                    model.save_segmentation(pred, str(output_path), colormap=False)
                    
            except Exception as e:
                logger.error(f"Error processing {img_path}: {str(e)}")
                continue
                
        logger.success(f"✓ Processing complete! Results saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error during model initialization or processing: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
        colorize=True
    )
    
    main()
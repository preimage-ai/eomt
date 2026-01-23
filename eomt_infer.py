#!/usr/bin/env python3
"""
EOMT Segmentation Inference with Feature Map Visualization
"""

import os
import sys
import importlib
import argparse
import yaml
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
        feature_map_dir: Optional[str] = None
    ):
        """
        Initialize EOMT segmentation model for inference
        
        Args:
            config_path: Path to EOMT config YAML file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
            save_feature_maps: Whether to save intermediate feature maps
            feature_map_dir: Directory to save feature maps
        """
        self.device = device
        self.img_size = (512, 1024)
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
            'models.vit.ViT': 'models.vit.ViT',
            'models.eomt.EoMT': 'models.eomt.EoMT',
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
        
        img_resized = img.resize((self.img_size[1], self.img_size[0]), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img_resized)).float()
        img_tensor = img_tensor.permute(2, 0, 1) / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)
        
        return img_tensor, orig_img, orig_size

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
            # Forward pass
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
            
            preds = per_pixel_logits.argmax(1)[0]
        
            min_conf = 0.1
            stuff_class_list = [1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            objects_mask = torch.tensor(stuff_class_list, device=self.device).bool() == 0
            objects_mask[51] = True # shed class priority
            objects_index_tensor = torch.arange(per_pixel_logits.shape[1], device=self.device)[objects_mask]
            per_pixel_objects_logits = per_pixel_logits[:, objects_mask, ...]
            has_objects = per_pixel_objects_logits[0].max(0)[0] > min_conf
            # not_void = preds != 165
            # has_objects = has_objects & not_void
            objects_selected = per_pixel_logits[:, objects_mask][:, :, has_objects].argmax(1)[0]
            preds[has_objects] = objects_index_tensor[objects_selected]

            preds = preds.cpu().numpy()

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
            feature_map_dir=args.feature_map_dir
        )
        
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
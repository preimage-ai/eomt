#!/usr/bin/env python3
"""
EOMT Segmentation Inference Script
Now works with authenticated HuggingFace DINOv3 access
"""

import argparse
import importlib
from pathlib import Path
from tqdm import tqdm
import yaml
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.amp.autocast_mode import autocast
from PIL import Image
from loguru import logger
import sys


class EOMTSegmentationInference:
    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda:0"):
        """
        Initialize EOMT segmentation model for inference
        
        Args:
            config_path: Path to EOMT config YAML file
            checkpoint_path: Path to model checkpoint
            device: Device to run inference on
        """
        self.device = device
        self.img_size = (512, 1024)

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        logger.info("Building EOMT model...")
        self._build_model_from_checkpoint(checkpoint_path)
        
        self.model.eval()
        logger.info(f"✓ Model initialized on {device}")
    
    def _fix_import_path(self, class_path: str):
        """Fix import paths from config"""
        replacements = {
            'models.vit.ViT': 'models.vit.ViT',
            'models.eomt.EoMT': 'models.eomt.EoMT',
        }
        return replacements.get(class_path, class_path)
    
    def _infer_num_classes_from_checkpoint(self, state_dict):
        """Infer number of classes from checkpoint"""
        for key in state_dict.keys():
            if 'class_head.weight' in key:
                checkpoint_classes = state_dict[key].shape[0]
                # EoMT adds +1 internally
                actual_num_classes = checkpoint_classes - 1
                logger.info(f"Checkpoint: {checkpoint_classes} output classes, using num_classes={actual_num_classes}")
                return actual_num_classes
        
        logger.warning("Could not infer num_classes, using 165")
        return 165
    
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
        
        # Build encoder (will download DINOv3 with your credentials)
        encoder_cfg = network_cfg["init_args"]["encoder"]
        encoder_path = self._fix_import_path(encoder_cfg["class_path"])
        encoder_module_name, encoder_class_name = encoder_path.rsplit(".", 1)
        
        encoder_module = importlib.import_module(encoder_module_name)
        ViT = getattr(encoder_module, encoder_class_name)
        
        encoder_init_args = encoder_cfg.get("init_args", {}).copy()
        
        logger.info("Building encoder (downloading DINOv3 with your HuggingFace credentials)...")
        # Don't pass pretrained=False - let it download normally
        encoder = ViT(img_size=self.img_size, **encoder_init_args)
        
        # Build EoMT network
        network_kwargs = {k: v for k, v in network_cfg["init_args"].items() if k != "encoder"}
        logger.info(f"Building EoMT with num_classes={num_classes}")
        
        # Create a simple wrapper since we don't have Lightning module
        from models.eomt import EoMT
        
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
        
        # It's OK to have some missing/unexpected for encoder since architectures might differ slightly
        if len(missing) > 300 or len(unexpected) > 300:
            logger.warning("Large number of mismatched keys - results may be poor")
            logger.warning(f"Sample missing: {missing[:5]}")
            logger.warning(f"Sample unexpected: {unexpected[:5]}")
    
    def preprocess_image(self, image_path: str):
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
    def predict(self, image_path: str):
        """Run segmentation inference"""
        img_tensor, orig_img, orig_size = self.preprocess_image(image_path)
        
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True, dtype=torch.float16):
            # Forward pass
            mask_logits_per_layer, class_logits_per_layer = self.model.network(img_tensor)
            
            # Use last layer
            mask_logits = F.interpolate(
                mask_logits_per_layer[-1], self.img_size, mode="bilinear"
            )
            
            # Convert to per-pixel logits
            B, Q, H, W = mask_logits.shape
            mask_probs = mask_logits.sigmoid()
            class_probs = class_logits_per_layer[-1].softmax(dim=-1)[..., :-1]  # Exclude void
            per_pixel_logits = torch.einsum('bqhw,bqc->bchw', mask_probs, class_probs)
            
            # Get predictions
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
    def apply_colormap(seg_mask: np.ndarray, num_classes: int = 165):
        """Apply colormap"""
        np.random.seed(42)
        colormap = np.random.randint(0, 255, (num_classes, 3), dtype=np.uint8)
        colormap[0] = [0, 0, 0]
        
        h, w = seg_mask.shape
        seg_colored = np.zeros((h, w, 3), dtype=np.uint8)
        max_class = int(seg_mask.max())
        for class_idx in range(min(num_classes, max_class + 1)):
            seg_colored[seg_mask == class_idx] = colormap[class_idx]
        
        return seg_colored


def main():
    parser = argparse.ArgumentParser(description="EOMT Segmentation Inference")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/dinov3/ade20k/semantic/eomt_large_512_synthetic.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--colormap", action="store_true")
    parser.add_argument("--save_raw", action="store_true")
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".jpeg", ".png", ".bmp"])
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    if args.colormap:
        (output_dir / "colored").mkdir(exist_ok=True)
    if args.save_raw:
        (output_dir / "raw").mkdir(exist_ok=True)
    
    logger.info("Initializing EOMT model...")
    try:
        model = EOMTSegmentationInference(
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            device=args.device
        )
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    image_files = []
    for ext in args.extensions:
        print("collecting images")
        image_files.extend(list(input_dir.rglob(f"*{ext}")))
        image_files.extend(list(input_dir.rglob(f"*{ext.upper()}")))
                           
    print("collected files")
    image_files = list(set(image_files))
    
    if not image_files:
        logger.error(f"No images found in {input_dir}")
        sys.exit(1)
    
    logger.info(f"Found {len(image_files)} images")
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            seg_mask = model.predict(str(img_path))
            output_name = img_path.relative_to(input_dir).stem
            
            if args.colormap:
                output_path = output_dir / "colored" / f"{output_name}_seg.png"
                model.save_segmentation(seg_mask, str(output_path), colormap=True)
            
            if args.save_raw:
                # np.save(str(output_dir / "raw" / f"{output_name}_seg.npy"), seg_mask)
                cv2.imwrite(str(output_dir / "raw" / f"{output_name}_seg.tif"), seg_mask)
            
            if not args.colormap and not args.save_raw:
                model.save_segmentation(seg_mask, str(output_dir / f"{output_name}_seg.png"), colormap=True)
                
        except Exception as e:
            logger.error(f"Error processing {img_path}: {e}")
            continue
    
    logger.info(f"✓ Complete! Results in {output_dir}")


if __name__ == "__main__":
    main()
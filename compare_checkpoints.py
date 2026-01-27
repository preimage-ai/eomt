#!/usr/bin/env python3
"""
Compare multiple EOMT checkpoints on a test dataset
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from tabulate import tabulate
import json

from eomt_infer import EOMTSegmentationInference


def compute_iou(pred, target, num_classes):
    """Compute per-class IoU"""
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls
        
        intersection = (pred_mask & target_mask).sum()
        union = (pred_mask | target_mask).sum()
        
        if union == 0:
            ious.append(float('nan'))
        else:
            ious.append(intersection / union)
    
    return ious


def evaluate_checkpoint(checkpoint_path, config_path, test_images, test_masks, device="cuda:0"):
    """Evaluate a single checkpoint"""
    print(f"\nEvaluating: {checkpoint_path}")
    
    model = EOMTSegmentationInference(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device
    )
    
    all_ious = []
    
    for img_path, mask_path in tqdm(zip(test_images, test_masks), total=len(test_images)):
        # Run inference
        pred = model.predict(str(img_path))
        
        # Load ground truth
        target = np.load(mask_path) if mask_path.suffix == '.npy' else np.array(Image.open(mask_path))
        
        # Compute IoU
        ious = compute_iou(pred, target, model.config.get('num_classes', 166))
        all_ious.append(ious)
    
    # Average IoU across all images
    mean_ious = np.nanmean(all_ious, axis=0)
    miou = np.nanmean(mean_ious)
    
    return {
        'checkpoint': Path(checkpoint_path).name,
        'miou': miou,
        'per_class_iou': mean_ious.tolist(),
        'num_images': len(test_images)
    }


def main():
    parser = argparse.ArgumentParser(description="Compare EOMT checkpoints")
    parser.add_argument("--checkpoints", nargs="+", required=True,
                       help="Paths to checkpoint files")
    parser.add_argument("--configs", nargs="+", required=True,
                       help="Paths to config files (same order as checkpoints)")
    parser.add_argument("--test_images", type=str, required=True,
                       help="Directory containing test images")
    parser.add_argument("--test_masks", type=str, required=True,
                       help="Directory containing test masks")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="Device to run inference on")
    parser.add_argument("--output", type=str, default="checkpoint_comparison.json",
                       help="Output JSON file for results")
    parser.add_argument("--extensions", nargs="+", default=[".jpg", ".png"],
                       help="Image file extensions")
    
    args = parser.parse_args()
    
    if len(args.checkpoints) != len(args.configs):
        print("Error: Number of checkpoints must match number of configs")
        return
    
    # Find test images
    test_images = []
    test_masks = []
    test_img_dir = Path(args.test_images)
    test_mask_dir = Path(args.test_masks)
    
    for ext in args.extensions:
        for img_path in test_img_dir.glob(f"*{ext}"):
            mask_path = test_mask_dir / f"{img_path.stem}.png"
            if mask_path.exists():
                test_images.append(img_path)
                test_masks.append(mask_path)
    
    print(f"Found {len(test_images)} test images")
    
    if len(test_images) == 0:
        print("No test images found!")
        return
    
    # Evaluate each checkpoint
    results = []
    for ckpt, config in zip(args.checkpoints, args.configs):
        result = evaluate_checkpoint(ckpt, config, test_images, test_masks, args.device)
        results.append(result)
    
    # Print comparison table
    table_data = []
    for result in results:
        table_data.append([
            result['checkpoint'],
            f"{result['miou']*100:.2f}%",
            result['num_images']
        ])
    
    print("\n" + "="*60)
    print("CHECKPOINT COMPARISON RESULTS")
    print("="*60)
    print(tabulate(table_data, headers=['Checkpoint', 'mIoU', 'Test Images'], tablefmt='grid'))
    
    # Find best checkpoint
    best_idx = max(range(len(results)), key=lambda i: results[i]['miou'])
    print(f"\n🏆 Best Checkpoint: {results[best_idx]['checkpoint']}")
    print(f"   mIoU: {results[best_idx]['miou']*100:.2f}%")
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {args.output}")


if __name__ == "__main__":
    from PIL import Image
    main()

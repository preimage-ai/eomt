#!/bin/bash
# Dual-Resolution Training from Scratch
# Usage: ./train_dual_resolution.sh <ade20k_zip_folder> <erp_images_dir> <erp_masks_dir>

ADE20K_PATH=$1
ERP_IMAGES=$2
ERP_MASKS=$3

if [ -z "$ADE20K_PATH" ] || [ -z "$ERP_IMAGES" ] || [ -z "$ERP_MASKS" ]; then
    echo "============================================================"
    echo "Dual-Resolution Training from Scratch"
    echo "============================================================"
    echo ""
    echo "Train a model from scratch on both perspective and ERP images"
    echo "with different resolutions in mixed batches."
    echo ""
    echo "Usage: ./train_dual_resolution.sh <ade20k_zip_folder> <erp_images_dir> <erp_masks_dir>"
    echo ""
    echo "Arguments:"
    echo "  ade20k_zip_folder  - Path to folder containing ADEChallengeData2016.zip"
    echo "  erp_images_dir     - Path to ERP images folder"
    echo "  erp_masks_dir      - Path to ERP masks folder"
    echo ""
    echo "Example:"
    echo "  ./train_dual_resolution.sh /data/ade20k /data/erp/images /data/erp/masks"
    echo ""
    echo "Configuration:"
    echo "  - Perspective resolution: 512×512"
    echo "  - ERP resolution: 1024×2048"
    echo "  - Batch composition: 95% perspective + 5% ERP (data-aware)"
    echo "  - Batch size: 16 (~15 perspective + ~1 ERP per batch)"
    echo "  - ERP augmentation: 200× (20 images → 4,000 samples)"
    echo "  - Epochs: 100"
    echo "  - Multi-scale ViT encoder"
    echo "  - Class weights enabled (rooftop classes masked for perspective)"
    echo ""
    echo "Requirements:"
    echo "  - ADEChallengeData2016.zip in ade20k_zip_folder"
    echo "  - ERP images (.jpg or .png) in erp_images_dir"
    echo "  - Corresponding masks (.png) in erp_masks_dir"
    echo "============================================================"
    exit 1
fi

# Validate ADE20K zip exists
if [ ! -f "$ADE20K_PATH/ADEChallengeData2016.zip" ]; then
    echo "ERROR: ADEChallengeData2016.zip not found in $ADE20K_PATH"
    echo "Please ensure the zip file exists at: $ADE20K_PATH/ADEChallengeData2016.zip"
    exit 1
fi

# Validate ERP directories exist
if [ ! -d "$ERP_IMAGES" ]; then
    echo "ERROR: ERP images directory not found: $ERP_IMAGES"
    exit 1
fi

if [ ! -d "$ERP_MASKS" ]; then
    echo "ERROR: ERP masks directory not found: $ERP_MASKS"
    exit 1
fi

echo ""
echo "========================================"
echo "Dual-Resolution Training from Scratch"
echo "========================================"
echo "Strategy: Train from scratch on mixed batches"
echo "Perspective: 512×512 (ADE20K ~21k images)"
echo "ERP: 1024×2048 (20 images × 200 aug = 4k samples)"
echo "Batch composition: 95% perspective + 5% ERP"
echo "Data-aware sampling (prevents ERP overfitting)"
echo "Multi-scale ViT with dynamic position embeddings"
echo "Epochs: 100"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - ADE20K Path: $ADE20K_PATH"
echo "  - ERP Images: $ERP_IMAGES"
echo "  - ERP Masks: $ERP_MASKS"
echo "  - Output: checkpoints/dual_resolution_scratch"
echo ""
echo "Starting training in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Execute training (using windowed ERP crops)
python main.py fit \
    --config configs/experiments/exp_dual_resolution_windowed.yaml \
    --data.init_args.perspective_path $ADE20K_PATH \
    --data.init_args.erp_image_dir $ERP_IMAGES \
    --data.init_args.erp_mask_dir $ERP_MASKS \
    --trainer.devices 1 \
    --trainer.precision "16-mixed"

echo ""
echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: checkpoints/dual_resolution_scratch"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Test on perspective images (512×512):"
echo "   python eomt_infer.py --checkpoint checkpoints/dual_resolution_scratch/eomt-best.ckpt"
echo ""
echo "2. Test on ERP images (1024×2048):"
echo "   python eomt_infer.py --checkpoint checkpoints/dual_resolution_scratch/eomt-best.ckpt"
echo ""
echo "3. Compare with other checkpoints:"
echo "   python compare_checkpoints.py"
echo ""
echo "Expected performance:"
echo "  - Perspective mIoU: 75-80%"
echo "  - ERP mIoU: 70-75%"
echo "========================================"

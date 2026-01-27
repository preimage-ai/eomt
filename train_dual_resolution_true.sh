#!/bin/bash
# ---------------------------------------------------------------
# True Dual-Resolution Training Script
# Perspective: 512×512 | ERP: 512×1024
# Mixed-size batches with custom collate function
# ---------------------------------------------------------------

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
if [ "$#" -lt 3 ]; then
    print_error "Usage: $0 <ade20k_path> <erp_images_dir> <erp_masks_dir>"
    echo ""
    echo "Arguments:"
    echo "  ade20k_path      : Path to ADE20K zip file"
    echo "  erp_images_dir   : Directory containing ERP images"
    echo "  erp_masks_dir    : Directory containing ERP masks"
    echo ""
    echo "Example:"
    echo "  $0 /data/ADEChallengeData2016.zip /data/erp/images /data/erp/masks"
    exit 1
fi

ADE20K_PATH="$1"
ERP_IMAGES="$2"
ERP_MASKS="$3"

# Validate paths
print_info "Validating input paths..."

if [ ! -f "$ADE20K_PATH" ]; then
    print_error "ADE20K zip file not found: $ADE20K_PATH"
    exit 1
fi

if [ ! -d "$ERP_IMAGES" ]; then
    print_error "ERP images directory not found: $ERP_IMAGES"
    exit 1
fi

if [ ! -d "$ERP_MASKS" ]; then
    print_error "ERP masks directory not found: $ERP_MASKS"
    exit 1
fi

print_info "All paths validated successfully!"

# Display experiment information
echo ""
echo "========================================"
echo "True Dual-Resolution Training"
echo "========================================"
echo "Strategy: Mixed-size batches"
echo "Perspective: 512×512 (ADE20K ~21k images)"
echo "ERP: 512×1024 (20 images × 100 aug = 2k samples)"
echo "Batch composition: 95% perspective + 5% ERP"
echo "Custom collate: Groups by size in batch"
echo "Epochs: 100"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - ADE20K Path: $ADE20K_PATH"
echo "  - ERP Images: $ERP_IMAGES"
echo "  - ERP Masks: $ERP_MASKS"
echo "  - Output: checkpoints/dual_resolution_true"
echo ""
echo "Key Features:"
echo "  ✓ True dual-resolution (no cropping/squishing)"
echo "  ✓ Perspective at native 512×512"
echo "  ✓ ERP at 512×1024 (preserves 1:2 aspect ratio)"
echo "  ✓ Dynamic ViT with position embedding interpolation"
echo "  ✓ Custom collate function for mixed batches"
echo ""
echo "Starting training in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Execute training
python main.py fit \
    --config configs/experiments/exp_dual_resolution_true.yaml \
    --data.init_args.perspective_path "$ADE20K_PATH" \
    --data.init_args.erp_image_dir "$ERP_IMAGES" \
    --data.init_args.erp_mask_dir "$ERP_MASKS" \
    --trainer.devices 1 \
    --trainer.precision "16-mixed"

echo ""
echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: checkpoints/dual_resolution_true"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Test on perspective images (512×512):"
echo "   python eomt_infer.py --input_dir /path/to/perspective \\"
echo "     --output_dir /path/to/output \\"
echo "     --checkpoint checkpoints/dual_resolution_true/eomt-best.ckpt \\"
echo "     --config configs/experiments/exp_dual_resolution_true.yaml \\"
echo "     --colormap"
echo ""
echo "2. Test on ERP images (512×1024 or larger) with windowing:"
echo "   python eomt_infer.py --input_dir /path/to/erp \\"
echo "     --output_dir /path/to/output \\"
echo "     --checkpoint checkpoints/dual_resolution_true/eomt-best.ckpt \\"
echo "     --config configs/experiments/exp_dual_resolution_true.yaml \\"
echo "     --use_windowed --window_size 512 \\"
echo "     --colormap"
echo ""
echo "3. Monitor training with TensorBoard:"
echo "   tensorboard --logdir checkpoints/dual_resolution_true"
echo ""

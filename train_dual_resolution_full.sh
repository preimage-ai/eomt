#!/bin/bash
# ---------------------------------------------------------------
# Full-Resolution Dual-Resolution Training Script
# Perspective: 1024×1024 (padded) | ERP: 1024×2048 (native)
# Optimized for 40GB GPU
# ---------------------------------------------------------------

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

print_highlight() {
    echo -e "${BLUE}[HIGHLIGHT]${NC} $1"
}

# Parse command line arguments
if [ "$#" -lt 3 ]; then
    print_error "Usage: $0 <ade20k_path> <erp_images_dir> <erp_masks_dir>"
    echo ""
    echo "Arguments:"
    echo "  ade20k_path      : Path to ADE20K zip file"
    echo "  erp_images_dir   : Directory containing ERP images (1024×2048)"
    echo "  erp_masks_dir    : Directory containing ERP masks (1024×2048)"
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
echo "Full-Resolution Dual-Resolution Training"
echo "========================================"
echo "Strategy: Native resolution training"
echo "Perspective: 1024×1024 (padded from 512×512)"
echo "ERP: 1024×2048 (native resolution)"
echo "Batch size: 4 (optimized for 40GB GPU)"
echo "Batch composition: 95% perspective + 5% ERP"
echo "Custom collate: Groups by size in batch"
echo "Epochs: 100"
echo "========================================"
echo ""
echo "Configuration:"
echo "  - ADE20K Path: $ADE20K_PATH"
echo "  - ERP Images: $ERP_IMAGES"
echo "  - ERP Masks: $ERP_MASKS"
echo "  - Output: checkpoints/dual_resolution_full"
echo ""
echo "Key Features:"
echo "  ✓ Full native resolution (1024×2048 for ERP)"
echo "  ✓ No cropping or squishing"
echo "  ✓ Maximum global context for panoramas"
echo "  ✓ Dynamic ViT with position embedding interpolation"
echo "  ✓ Gradient clipping for training stability"
echo "  ✓ Mixed precision (fp16) for memory efficiency"
echo ""
echo "GPU Requirements:"
echo "  - Minimum: 40GB VRAM"
echo "  - Recommended: A100 40GB or better"
echo "  - Batch size: 4 (3 perspective + 1 ERP typical)"
echo ""
print_highlight "Memory Usage Estimate:"
echo "  - 1024×1024 image: ~6-8 GB per batch"
echo "  - 1024×2048 image: ~12-16 GB per batch"
echo "  - Total: ~24-32 GB with overhead"
echo ""
print_warning "This will use significant GPU memory!"
echo ""
echo "Starting training in 5 seconds... (Ctrl+C to cancel)"
sleep 5

# Execute training
python main.py fit \
    --config configs/experiments/exp_dual_resolution_full.yaml \
    --data.init_args.perspective_path "$ADE20K_PATH" \
    --data.init_args.erp_image_dir "$ERP_IMAGES" \
    --data.init_args.erp_mask_dir "$ERP_MASKS" \
    --trainer.devices 1 \
    --trainer.precision "16-mixed"

echo ""
echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: checkpoints/dual_resolution_full"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Test on perspective images (any size):"
echo "   python eomt_infer.py --input_dir /path/to/perspective \\"
echo "     --output_dir /path/to/output \\"
echo "     --checkpoint checkpoints/dual_resolution_full/eomt-best.ckpt \\"
echo "     --config configs/experiments/exp_dual_resolution_full.yaml \\"
echo "     --colormap"
echo ""
echo "2. Test on ERP images at native 1024×2048 (direct inference):"
echo "   python eomt_infer.py --input_dir /path/to/erp \\"
echo "     --output_dir /path/to/output \\"
echo "     --checkpoint checkpoints/dual_resolution_full/eomt-best.ckpt \\"
echo "     --config configs/experiments/exp_dual_resolution_full.yaml \\"
echo "     --colormap"
echo ""
echo "3. Test on larger ERP images (e.g., 2048×4096) with windowing:"
echo "   python eomt_infer.py --input_dir /path/to/large_erp \\"
echo "     --output_dir /path/to/output \\"
echo "     --checkpoint checkpoints/dual_resolution_full/eomt-best.ckpt \\"
echo "     --config configs/experiments/exp_dual_resolution_full.yaml \\"
echo "     --use_windowed --window_size 1024 \\"
echo "     --colormap"
echo ""
echo "4. Monitor training with TensorBoard:"
echo "   tensorboard --logdir checkpoints/dual_resolution_full"
echo ""
print_highlight "Benefits of this approach:"
echo "  ✓ Maximum quality for ERP segmentation"
echo "  ✓ Full panorama context (no cropping)"
echo "  ✓ Direct inference for 1024×2048 images"
echo "  ✓ Automatic windowing for larger images"
echo ""

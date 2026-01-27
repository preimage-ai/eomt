#!/bin/bash
# Cubemap-to-ERP Fine-tuning Script
# Usage: ./train_cubemap_to_erp.sh <option> <erp_images_dir> <erp_masks_dir> [checkpoint_path]

OPTION=$1
ERP_IMAGES=$2
ERP_MASKS=$3
CHECKPOINT=${4:-eomt-epoch=015.ckpt}

if [ -z "$OPTION" ] || [ -z "$ERP_IMAGES" ] || [ -z "$ERP_MASKS" ]; then
    echo "============================================================"
    echo "Cubemap-to-ERP Fine-tuning"
    echo "============================================================"
    echo ""
    echo "Your checkpoint is trained on cubemaps from 20 ERP images."
    echo "This script fine-tunes it on the full ERP format."
    echo ""
    echo "Usage: ./train_cubemap_to_erp.sh <option> <erp_images_dir> <erp_masks_dir> [checkpoint]"
    echo ""
    echo "Options:"
    echo "  1 - Direct ERP fine-tuning (RECOMMENDED, 30x aug, 40 epochs)"
    echo "  2 - Gradual resolution adaptation (SAFEST, phased training, 50 epochs)"
    echo "  3 - Multi-scale ViT (FLEXIBLE, handles any resolution, 40 epochs)"
    echo ""
    echo "Examples:"
    echo "  ./train_cubemap_to_erp.sh 1 /data/erp/images /data/erp/masks"
    echo "  ./train_cubemap_to_erp.sh 2 /data/erp/images /data/erp/masks eomt-epoch=015.ckpt"
    echo ""
    echo "Default checkpoint: eomt-epoch=015.ckpt"
    echo "============================================================"
    exit 1
fi

case $OPTION in
    1)
        CONFIG="configs/experiments/CUBEMAP_exp1_erp_direct.yaml"
        NAME="cubemap_to_erp_direct"
        LR="5e-6"
        echo ""
        echo "========================================"
        echo "Option 1: Direct ERP Fine-tuning"
        echo "========================================"
        echo "Strategy: 30x augmentation, direct adaptation"
        echo "Resolution: (512, 1024) from start"
        echo "Epochs: 40"
        echo "Learning Rate: 5e-6"
        echo "Best for: Quick adaptation, recommended"
        echo "========================================"
        ;;
    2)
        CONFIG="configs/experiments/CUBEMAP_exp2_gradual_adaptation.yaml"
        NAME="cubemap_to_erp_gradual"
        LR="5e-6"
        echo ""
        echo "========================================"
        echo "Option 2: Gradual Resolution Adaptation"
        echo "========================================"
        echo "Strategy: Phased training"
        echo "  Phase 1 (10 epochs): (512, 512)"
        echo "  Phase 2 (20 epochs): (512, 768)"
        echo "  Phase 3 (20 epochs): (512, 1024)"
        echo "Learning Rate: 5e-6"
        echo "Best for: Safest adaptation, highest performance"
        echo "========================================"
        ;;
    3)
        CONFIG="configs/experiments/CUBEMAP_exp3_multiscale.yaml"
        NAME="cubemap_to_erp_multiscale"
        LR="5e-6"
        echo ""
        echo "========================================"
        echo "Option 3: Multi-Scale ViT"
        echo "========================================"
        echo "Strategy: Multi-scale position embeddings"
        echo "Resolution: Mixed (512,512) to (512,1024)"
        echo "Epochs: 40"
        echo "Learning Rate: 5e-6"
        echo "Best for: Maximum flexibility"
        echo "========================================"
        ;;
    *)
        echo "Invalid option: $OPTION"
        echo "Choose 1, 2, or 3"
        exit 1
        ;;
esac

echo ""
echo "Configuration:"
echo "  - ERP Images: $ERP_IMAGES"
echo "  - ERP Masks: $ERP_MASKS"
echo "  - Checkpoint: $CHECKPOINT"
echo "  - Output: checkpoints/$NAME"
echo "  - Learning Rate: $LR"
echo ""
echo "Starting training in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Build and execute command
python main.py fit \
    --config $CONFIG \
    --data.init_args.erp_image_dir $ERP_IMAGES \
    --data.init_args.erp_mask_dir $ERP_MASKS \
    --model.init_args.ckpt_path $CHECKPOINT \
    --optimizer.init_args.lr $LR \
    --trainer.callbacks.dirpath checkpoints/$NAME \
    --trainer.devices 1 \
    --trainer.precision "16-mixed"

echo ""
echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: checkpoints/$NAME"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Test inference:"
echo "   python eomt_infer.py --checkpoint checkpoints/$NAME/eomt-best.ckpt"
echo ""
echo "2. Compare with original:"
echo "   python compare_checkpoints.py --checkpoints $CHECKPOINT checkpoints/$NAME/eomt-best.ckpt"
echo ""
echo "Expected improvement: 65-70% -> 75-82% mIoU on ERP"
echo "========================================"

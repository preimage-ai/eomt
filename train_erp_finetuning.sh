#!/bin/bash
# ERP Fine-tuning Script for Limited Data (20 ERP images)
# Usage: ./train_erp_finetuning.sh <option> <erp_images_dir> <erp_masks_dir> [ade20k_path] [checkpoint_path]

OPTION=$1
ERP_IMAGES=$2
ERP_MASKS=$3
ADE20K_PATH=$4
CHECKPOINT=${5:-eomt-epoch=015.ckpt}

if [ -z "$OPTION" ] || [ -z "$ERP_IMAGES" ] || [ -z "$ERP_MASKS" ]; then
    echo "============================================================"
    echo "ERP Fine-tuning for Limited Data (20 images)"
    echo "============================================================"
    echo ""
    echo "Usage: ./train_erp_finetuning.sh <option> <erp_images_dir> <erp_masks_dir> [ade20k_path] [checkpoint]"
    echo ""
    echo "Options:"
    echo "  A - Pure ERP fine-tuning (aggressive, 100x augmentation)"
    echo "  B - Mixed fine-tuning (RECOMMENDED, 50x aug + 70% perspective)"
    echo "  C - Position embedding adaptation (advanced, multi-scale)"
    echo ""
    echo "Examples:"
    echo "  ./train_erp_finetuning.sh B /data/erp/images /data/erp/masks /data/ade20k"
    echo "  ./train_erp_finetuning.sh A /data/erp/images /data/erp/masks"
    echo ""
    echo "Default checkpoint: eomt-epoch=015.ckpt"
    echo "============================================================"
    exit 1
fi

case $OPTION in
    A)
        CONFIG="configs/experiments/REVISED_exp2_erp_finetune_pure.yaml"
        NAME="erp_pure"
        NEEDS_ADE20K=0
        echo ""
        echo "========================================"
        echo "Option A: Pure ERP Fine-tuning"
        echo "========================================"
        echo "Strategy: 100x augmentation, pure ERP"
        echo "Risk: May forget perspective knowledge"
        echo "Best for: Pure ERP deployment"
        echo "========================================"
        ;;
    B)
        CONFIG="configs/experiments/REVISED_exp3_erp_finetune_mixed.yaml"
        NAME="erp_mixed"
        NEEDS_ADE20K=1
        echo ""
        echo "========================================"
        echo "Option B: Mixed Fine-tuning (RECOMMENDED)"
        echo "========================================"
        echo "Strategy: 50x aug + 70% perspective mix"
        echo "Risk: Low - maintains perspective perf"
        echo "Best for: Production deployment"
        echo "========================================"
        ;;
    C)
        CONFIG="configs/experiments/REVISED_exp4_position_embed_adaptation.yaml"
        NAME="erp_adaptive"
        NEEDS_ADE20K=1
        echo ""
        echo "========================================"
        echo "Option C: Position Embedding Adaptation"
        echo "========================================"
        echo "Strategy: Multi-scale ViT + 50% mix"
        echo "Risk: Medium - most complex"
        echo "Best for: Maximum flexibility"
        echo "========================================"
        ;;
    *)
        echo "Invalid option: $OPTION"
        echo "Choose A, B, or C"
        exit 1
        ;;
esac

# Check if ADE20K path is required and provided
if [ $NEEDS_ADE20K -eq 1 ]; then
    if [ -z "$ADE20K_PATH" ]; then
        echo "ERROR: Option $OPTION requires ADE20K path for perspective mixing"
        echo "Usage: ./train_erp_finetuning.sh $OPTION $ERP_IMAGES $ERP_MASKS <ade20k_path>"
        exit 1
    fi
fi

echo ""
echo "Configuration:"
echo "  - ERP Images: $ERP_IMAGES"
echo "  - ERP Masks: $ERP_MASKS"
if [ $NEEDS_ADE20K -eq 1 ]; then
    echo "  - ADE20K Path: $ADE20K_PATH"
fi
echo "  - Checkpoint: $CHECKPOINT"
echo "  - Output: checkpoints/$NAME"
echo ""
echo "Starting training in 3 seconds... (Ctrl+C to cancel)"
sleep 3

# Build command
CMD="python main.py fit --config $CONFIG --data.init_args.erp_image_dir $ERP_IMAGES --data.init_args.erp_mask_dir $ERP_MASKS --model.init_args.ckpt_path $CHECKPOINT --trainer.callbacks.dirpath checkpoints/$NAME --trainer.devices 1 --trainer.precision \"16-mixed\""

# Add ADE20K path if needed
if [ $NEEDS_ADE20K -eq 1 ]; then
    CMD="$CMD --data.init_args.perspective_path $ADE20K_PATH"
fi

# Execute
echo ""
echo "Executing: $CMD"
echo ""
eval $CMD

echo ""
echo "========================================"
echo "Training complete!"
echo "Checkpoints saved to: checkpoints/$NAME"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Test with: python eomt_infer.py --checkpoint checkpoints/$NAME/eomt-best.ckpt"
echo "2. Compare: python compare_checkpoints.py"
echo "========================================"

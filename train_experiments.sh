#!/bin/bash
# Training script for EOMT experiments
# Usage: ./train_experiments.sh <experiment_number> <data_path>

EXPERIMENT=$1
DATA_PATH=$2

if [ -z "$EXPERIMENT" ] || [ -z "$DATA_PATH" ]; then
    echo "Usage: ./train_experiments.sh <experiment_number> <data_path>"
    echo "Example: ./train_experiments.sh 4 /path/to/ade20k"
    echo ""
    echo "Available experiments:"
    echo "  1 - ERP Only (512x1024)"
    echo "  2 - ERP with Higher Queries (200)"
    echo "  3 - Perspective Only (512x512)"
    echo "  4 - Mixed Resolution (RECOMMENDED)"
    echo "  5 - Multi-Scale ViT (ADVANCED)"
    exit 1
fi

case $EXPERIMENT in
    1)
        CONFIG="configs/experiments/exp1_erp_only_512x1024.yaml"
        NAME="exp1_erp_only"
        ;;
    2)
        CONFIG="configs/experiments/exp2_erp_higher_queries.yaml"
        NAME="exp2_erp_higher_queries"
        ;;
    3)
        CONFIG="configs/experiments/exp3_perspective_only_512x512.yaml"
        NAME="exp3_perspective_only"
        ;;
    4)
        CONFIG="configs/experiments/exp4_mixed_resolution.yaml"
        NAME="exp4_mixed_resolution"
        ;;
    5)
        CONFIG="configs/experiments/exp5_multiscale_vit.yaml"
        NAME="exp5_multiscale_vit"
        ;;
    *)
        echo "Invalid experiment number: $EXPERIMENT"
        echo "Choose 1-5"
        exit 1
        ;;
esac

echo "=========================================="
echo "Starting Experiment $EXPERIMENT: $NAME"
echo "Config: $CONFIG"
echo "Data Path: $DATA_PATH"
echo "=========================================="

python main.py fit \
    --config $CONFIG \
    --data.init_args.path $DATA_PATH \
    --trainer.devices 1 \
    --trainer.precision "16-mixed" \
    --trainer.callbacks.dirpath "checkpoints/$NAME" \
    --trainer.callbacks.filename "eomt-{epoch:03d}"

echo "=========================================="
echo "Training complete for Experiment $EXPERIMENT"
echo "Checkpoints saved to: checkpoints/$NAME"
echo "=========================================="

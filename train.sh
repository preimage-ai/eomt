conda activate eomt
python main.py fit -c configs/dinov3/ade20k/semantic/eomt_large_512.yaml --data.path solar2 --ckpt_path checkpoints/last.ckpt

# python3 main.py fit \
#   -c configs/dinov2/coco/panoptic/eomt_large_640.yaml \
#   --trainer.devices 4 \
#   --data.batch_size 4 \
#   --data.path /path/to/dataset

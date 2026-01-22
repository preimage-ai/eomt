python eomt_infer.py \
    --input_dir ~/Workspace/data/solar/nagpur_stub/LMH109760/out/sharp_out/ \
    --output_dir fout2 \
    --config configs/eomt_large_512.yaml \
    --checkpoint eomt_class_bal4_12.ckpt \
    --save_raw \
    --colormap \
    --save_feature_maps \
    --feature_map_dir feature_maps \
    --extensions .jpg .png
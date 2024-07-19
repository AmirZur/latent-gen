python das_eval.py \
    --model_name_or_path "inst_tune" \
    --das_path "das/inst_tune/2024-07-19-10-37-53/weights" \
    --positions "l1" \
    --share_weights \
    --intervention_offset 0 \
    --dataset_split "train_inclusive" \
    --output_dir "das_eval"
# greedy sampling
python das_eval.py \
    --model_name_or_path "inst_tune" \
    --das_path $1 \
    --positions "l1" \
    --share_weights \
    --intervention_offset $2 \
    --dataset_split "train_inclusive" \
    --output_dir "das_eval"

# random sampling
python das_eval.py \
    --model_name_or_path "inst_tune" \
    --das_path $1 \
    --positions "l1" \
    --share_weights \
    --intervention_offset $2 \
    --do_sample \
    --temperature 0.7 \
    --dataset_split "train_inclusive" \
    --output_dir "das_eval"
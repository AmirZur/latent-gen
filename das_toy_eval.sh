python das_eval.py \
    --model_name_or_path "gpt2" \
    --das_path "das/gpt2/2024-07-20-11-11-06/weights" \
    --toy_dataset \
    --positions "l1" \
    --share_weights \
    --intervention_offset 0 \
    --output_dir "das_eval" \
    --max_new_tokens 8 \
    --batch_size 8
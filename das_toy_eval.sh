python das_eval.py \
    --model_name_or_path "gpt2" \
    --das_path "das/gpt2/2024-07-22-12-27-25/weights" \
    --toy_dataset \
    --positions "f4" \
    --share_weights \
    --intervention_offset 0 \
    --output_dir "das_eval" \
    --max_new_tokens 8 \
    --batch_size 8
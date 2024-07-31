python probe.py \
    --model_name_or_path "inst_tune/train" \
    --use_flash_attention \
    --output_dir "inst_tune/probe" \
    --num_tokens_from_end 10
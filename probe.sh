python probe.py \
    --model_name_or_path "inst_tune/train" \
    --use_flash_attention \
    --train_path "inst_tune/vanilla_train/eval/generations.csv" \
    --validation_path "inst_tune/vanilla_valid/eval/generations.csv" \
    --output_dir "inst_tune/probe_vanilla" \
    --num_tokens_from_end 10
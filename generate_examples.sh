python generate_examples.py \
    --model_name_or_path "inst_tune" \
    --output_dir "inst_gens" \
    --use_flash_attention \
    --do_sample \
    --temperature 0.7

python generate_examples.py \
    --model_name_or_path "inst_tune" \
    --output_dir "inst_gens" \
    --use_flash_attention \
    --do_sample \
    --temperature 0.7 \
    --prefill_generation
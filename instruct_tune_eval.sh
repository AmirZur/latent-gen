for split in "validation" "train_inclusive"
do
    python instruct_tune.py \
        --do_eval \
        --do_classify \
        --model_name_or_path "inst_tune/train" \
        --use_flash_attention \
        --prompt_with_example \
        --eval_split $split \
        --output_dir "inst_tune" \
        --per_device_eval_batch_size 8 \
        --aspect "service" \
        --max_new_tokens 256 \
        --num_return_sequences 10 \
        --do_sample \
        --temperature 0.7
done
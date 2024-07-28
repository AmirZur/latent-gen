python instruct_tune.py \
    --model_name_or_path "microsoft/Phi-3-mini-4k-instruct" \
    --use_flash_attention \
    --prompt_with_example \
    --output_dir "inst_tune" \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --logging_dir "logs" \
    --logging_steps 10 \
    --max_seq_length 256 \
    --save_model \
    --use_wandb
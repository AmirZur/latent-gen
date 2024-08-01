python instruct_tune.py \
    --do_train \
    --do_eval \
    --do_classify \
    --model_name_or_path "microsoft/Phi-3-mini-4k-instruct" \
    --use_flash_attention \
    --one_shot \
    --num_replicas 10 \
    --output_dir "inst_tune" \
    --per_device_train_batch_size 32 \
    --num_train_epochs 5 \
    --learning_rate 1e-5 \
    --eval_split "validation" \
    --per_device_eval_batch_size 8 \
    --aspect "service" \
    --max_new_tokens 256 \
    --num_return_sequences 10 \
    --do_sample \
    --temperature 0.7 \
    --logging_dir "logs" \
    --logging_steps 10 \
    --max_seq_length 256 \
    --save_model \
    --use_wandb
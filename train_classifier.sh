python train_classifier.py \
    --model_name_or_path "meta-llama/Llama-2-7b-hf" \
    --dataset_split "train_inclusive" \
    --output_dir "train_classifier" \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --logging_dir "logs" \
    --logging_steps 1 \
    --max_seq_length 256 \
    --eval_steps 10 \
    --save_model
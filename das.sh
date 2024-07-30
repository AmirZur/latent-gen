for layer in 20 24 28 32
do
    python das.py \
        --model_name_or_path "inst_tune/train" \
        --use_flash_attention \
        --layers "$layer" \
        --subspace_dim 128 \
        --positions "l1" \
        --share_weights \
        --dropout 0 \
        --intervention_offset 0 \
        --prompt_with_example \
        --num_train_epochs 5 \
        --learning_rate 1e-3 \
        --batch_size 8 \
        --max_seq_length 256 \
        --max_new_tokens 256 \
        --do_sample \
        --temperature 0.7 \
        --logging_steps 10 \
        --output_dir "das" \
        --save_model \
        --use_wandb
done
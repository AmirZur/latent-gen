for i in 0 5 10 15 20
do
    python das.py \
        --model_name_or_path "inst_tune" \
        --layers "20;24" \
        --subspace_dim 128 \
        --positions "l1" \
        --intervention_offset $i \
        --share_weights \
        --dropout 0 \
        --num_train_epochs 10 \
        --learning_rate 1e-3 \
        --batch_size 8 \
        --max_seq_length 256 \
        --logging_steps 10 \
        --output_dir=das \
        --use_wandb
done
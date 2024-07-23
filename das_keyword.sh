# match first instance of keyword
python das.py \
    --model_name_or_path "inst_tune" \
    --layers "20;24" \
    --subspace_dim 128 \
    --positions "l1" \
    --intervention_offset 0 \
    --binary \
    --keyword_match \
    --keyword_match_first \
    --share_weights \
    --dropout 0 \
    --num_train_epochs 10 \
    --learning_rate 1e-3 \
    --batch_size 8 \
    --max_seq_length 256 \
    --logging_steps 10 \
    --output_dir=das \
    --use_wandb \
    --save_model

# match last instance of keyword
python das.py \
    --model_name_or_path "inst_tune" \
    --layers "20;24" \
    --subspace_dim 128 \
    --positions "l1" \
    --intervention_offset 0 \
    --binary \
    --keyword_match \
    --share_weights \
    --dropout 0 \
    --num_train_epochs 10 \
    --learning_rate 1e-3 \
    --batch_size 8 \
    --max_seq_length 256 \
    --logging_steps 10 \
    --output_dir=das \
    --use_wandb \
    --save_model
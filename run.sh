
# data preparation

python3 custom_data/data_prep.py \
--source_data_dir data/train \
--output_data_dir output_data/train

python3 custom_data/data_prep.py \
--source_data_dir data/dev \
--output_data_dir output_data/dev

python3 custom_data/data_prep.py \
--source_data_dir data/test \
--output_data_dir output_data/test

# finetuning

torchrun --nproc_per_node=10 train/fine-tune_on_custom_dataset.py \
--model_name openai/whisper-small \
--language zh \
--sampling_rate 16000 \
--num_proc 4 \
--train_strategy epoch \
--learning_rate 3e-3 \
--warmup 1000 \
--train_batchsize 8 \
--eval_batchsize 4 \
--num_epochs 20 \
--resume_from_ckpt None \
--output_dir op_dir_epoch \
--train_datasets output_data/train  \
--eval_datasets output_data/dev output_data/test

# evaluation

python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch/checkpoint-394" \
--temp_ckpt_folder "temp" \
--language zh \
--eval_datasets output_data/dev output_data/test \
--device 0 \
--batch_size 8 \
--output_dir predictions_dir


# data preparation for pinyin

python3 custom_data/data_prep.py \
--source_data_dir data-pinyin/train \
--output_data_dir output_data-pinyin/train

python3 custom_data/data_prep.py \
--source_data_dir data-pinyin/dev \
--output_data_dir output_data-pinyin/dev

python3 custom_data/data_prep.py \
--source_data_dir data-pinyin/test \
--output_data_dir output_data-pinyin/test

# data preparation for character

python3 custom_data/data_prep.py \
--source_data_dir data-char/train \
--output_data_dir output_data-char/train

python3 custom_data/data_prep.py \
--source_data_dir data-char/dev \
--output_data_dir output_data-char/dev

python3 custom_data/data_prep.py \
--source_data_dir data-char/test \
--output_data_dir output_data-char/test

# finetuning pinyin

torchrun --nproc_per_node=10 train/fine-tune_on_custom_dataset.py \
--model_name openai/whisper-medium \
--language en \
--sampling_rate 16000 \
--num_proc 4 \
--train_strategy epoch \
--learning_rate 6.25e-6 \
--warmup 1000 \
--train_batchsize 12 \
--eval_batchsize 6 \
--num_epochs 20 \
--resume_from_ckpt None \
--output_dir op_dir_epoch-pinyin \
--train_datasets output_data-pinyin/train  \
--eval_datasets output_data-pinyin/dev

# evaluation pinyin

python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch-pinyin/checkpoint-394" \
--temp_ckpt_folder "temp-pinyin" \
--language en \
--eval_datasets output_data-pinyin/test \
--device 0 \
--batch_size 8 \
--output_dir predictions_dir-pinyin

# finetuning char

torchrun --nproc_per_node=10 train/fine-tune_on_custom_dataset.py \
--model_name openai/whisper-medium \
--language zh \
--sampling_rate 16000 \
--num_proc 4 \
--train_strategy epoch \
--learning_rate 6.25e-6 \
--warmup 1000 \
--train_batchsize 12 \
--eval_batchsize 6 \
--num_epochs 20 \
--resume_from_ckpt None \
--output_dir op_dir_epoch-char \
--train_datasets output_data-char/train  \
--eval_datasets output_data-char/dev

# evaluation char

python3 evaluate/evaluate_on_custom_dataset.py \
--is_public_repo False \
--ckpt_dir "op_dir_epoch-char/checkpoint-394" \
--temp_ckpt_folder "temp-char" \
--language zh \
--eval_datasets output_data-char/test \
--device 0 \
--batch_size 8 \
--output_dir predictions_dir-char

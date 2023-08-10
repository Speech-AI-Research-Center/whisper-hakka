torchrun --nproc_per_node=1 train/fine-tune_on_custom_dataset.py \
<<<<<<< HEAD
--model_name openai/whisper-base \
=======
--model_name openai/whisper-medium \
>>>>>>> 609c286cd0a98a27eddc40796de23099b776a7f0
--language zh \
--sampling_rate 16000 \
--num_proc 4 \
--train_strategy epoch \
--learning_rate 6.25e-6 \
--warmup 1000 \
<<<<<<< HEAD
--train_batchsize 16 \
--eval_batchsize 8 \
--num_epochs 100 \
--resume_from_ckpt None \
--output_dir op_dir_epoch-char \
--train_datasets output_data-char/train  \
--eval_datasets output_data-char/dev
=======
--train_batchsize 12 \
--eval_batchsize 6 \
--num_epochs 20 \
--resume_from_ckpt None \
--output_dir op_dir_epoch-han \
--train_datasets output_data-han/train  \
--eval_datasets output_data-han/dev

>>>>>>> 609c286cd0a98a27eddc40796de23099b776a7f0

#!/bin/sh

conda activate myenv

#E-Context
#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2 -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 250 -batch_size 50 -train_steps 2800 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 500 -warmup_steps_dec 300 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/E_Contect -train_from ../models/models/model_step_148000.pt -use_dec_lr -use_bert_lr

#D-Context
#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2 -sep_optim true -lr_bert 0.00002 -lr_dec 0.02 -save_checkpoint_steps 250 -batch_size 50 -train_steps 2800 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 500 -warmup_steps_dec 300 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/E_Contect -train_from ../models/models/model_step_148000.pt -use_dec_lr -use_bert_lr

#A-Context
#python train.py  -task abs -mode train -bert_data_path ../bert_data/abs/bert_data -dec_dropout 0.2 -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 250 -batch_size 50 -train_steps 2800 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 500 -warmup_steps_dec 300 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/E_Contect -train_from ../models/models/model_step_148000.pt -use_dec_lr -use_bert_lr


#!/bin/sh

conda activate myenv


#Todo: try rand_doc, doc with two different hyper parameter configa
##################
#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.0002 -lr_dec 0.02 -save_checkpoint_steps 250 -batch_size 50 -train_steps 2800 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 300 -warmup_steps_dec 300 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/ENC0002_DEC_02/ -train_from ../models/models/model_step_148000.pt -use_dec_lr -use_bert_lr

#python train.py -task abs -mode validate -batch_size 25 -test_batch_size 100 -bert_data_path ../bert_data/deterministic/bert_data -log_file ../logs/EVAL_ALL_RAND -model_path ../models/ENC0002_DEC_02/ -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 5 -result_path ../logs/EVAL_ALL_RESULTS_NEW_RAND

#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.00002 -lr_dec 0.02 -save_checkpoint_steps 250 -batch_size 50 -train_steps 2800 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 300 -warmup_steps_dec 300 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/ENC00002_DEC_02/ -train_from ../models/models/model_step_148000.pt -use_dec_lr -use_bert_lr

#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 250 -batch_size 50 -train_steps 3000 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 300 -warmup_steps_dec 300 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/ENC_002_DEC_02/ -train_from ../models/models/model_step_148000.pt -use_dec_lr  -use_bert_lr


#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 400 -batch_size 50 -train_steps 8000 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/ENC002_DEC02_10K_WARMUP/ -train_from ../models/models/model_step_148000.pt -use_dec_lr -use_bert_lr

#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.00002 -lr_dec 0.02 -save_checkpoint_steps 400 -batch_size 50 -train_steps 2500 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 500 -warmup_steps_dec 300 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/ENC00002_DEC02_500_300_WARMUP/ -train_from ../models/models/model_step_148000.pt -use_dec_lr -use_bert_lr

#rm ../models/ENC00002_DEC02_500_300_WARMUP/model_step_1*

#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.00002 -lr_dec 0.02 -save_checkpoint_steps 400 -batch_size 50 -train_steps 2500 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 500 -warmup_steps_dec 500 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/ENC00002_DEC02_500_500_WARMUP/ -train_from ../models/models/model_step_148000.pt -use_dec_lr -use_bert_lr

#rm ../models/ENC00002_DEC02_500_500_WARMUP/model_step_1*

#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 250 -batch_size 50 -train_steps 2500 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 500 -warmup_steps_dec 500 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/ENC_002_DEC_02_500_500_WARMUP/ -train_from ../models/models/model_step_148000.pt -use_dec_lr  -use_bert_lr

#rm ../models/ENC_002_DEC02_500_500_WARMUP/model_step_1*

#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 400 -batch_size 50 -train_steps 2500 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 500 -warmup_steps_dec 300 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/ENC_002_DEC_02_500_300_WARMUP/ -train_from ../models/models/model_step_148000.pt -use_dec_lr  -use_bert_lr

#for folder in ../models/*WARMUP/;
#do
#echo $folder
#python train.py -task abs -mode validate -batch_size 25 -test_batch_size 100 -bert_data_path ../bert_data/deterministic/bert_data -log_file ../logs/EVAL_ALL_RAND -model_path $folder -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 5 -result_path ../logs/EVAL_ALL_RESULTS_NEW_RAND
#done

#apython train.py -task abs -mode validate -batch_size 25 -test_batch_size 500 -bert_data_path ../bert_data/deterministic/bert_data -log_file ../logs/EVAL_ALL_RAND -model_path ../models/ENC_002_DEC_02/ -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 5 -result_path ../logs/EVAL_ALL_RESULTS_NEW_RAND
##################


#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.0 -lr_dec 0.02 -save_checkpoint_steps 250 -batch_size 50 -train_steps 2800 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 300 -warmup_steps_dec 300 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/ENC0_DEC_02/ -train_from ../models/models/model_step_148000.pt -use_dec_lr -use_bert_lr



#python train.py -task abs -mode validate -batch_size 25 -test_batch_size 100 -bert_data_path ../bert_data/deterministic/bert_data -log_file ../logs/EVAL_ALL_RAND -model_path ../models/ENC0_DEC_02/ -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 5 -result_path ../logs/EVAL_ALL_RESULTS_NEW_RAND



#python train.py  -task abs -mode train -bert_data_path ../bert_data/unfun/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.00002 -lr_dec 0.02 -save_checkpoint_steps 20 -batch_size 50 -train_steps 500 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 500 -warmup_steps_dec 500 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/unfun_ENC00002_DEC02_W500/ -train_from ../models/models/model_step_148000.pt -use_dec_lr  -use_bert_lr

python train.py -task abs -mode validate -batch_size 25 -test_batch_size 100 -bert_data_path ../bert_data/unfun/bert_data -log_file ../logs/EVAL_ALL_RAND -model_path ../models/unfun_ENC00002_DEC02_W500/  -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 5 -result_path ../logs/EVAL_ALL_RESULTS_NEW_RAND

#Finetune #1  #was batches of 200 250, now 400!
#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 200 -batch_size 50 -train_steps 156000 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/CLEAN_12K_LR02_CONT/ -train_from ../models/CLEAN_12K_LR02_CONT/model_step_154000.pt # ../models/models/model_step_148000.pt #-use_dec_lr
#python train.py  -task abs -mode train -bert_data_path ../bert_data/abs/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.002 -lr_dec 0.02 -save_checkpoint_steps 250 -batch_size 50 -train_steps 2800 -report_every 50 -accum_count 4 -use_bert_emb true -use_interval true -warmup_steps_bert 300 -warmup_steps_dec 300 -max_pos 512 -visible_gpus 0 -log_file fine_tune_generator -model_path ../models/ENC_ABS_10xDEC_12K_LR02/ -train_from ../models/ENC_ABS_10xDEC_12K_LR02/model_step_2500.pt #models/model_step_148000.pt -use_dec_lr -use_bert_lr
#Next, try training encoder with higher learning rate

#Dec lr was 0.001


#Finetune #1.5
#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 350 -batch_size 10 -train_steps 157000 -report_every 50 -accum_count 1 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0 -log_file ../logs/raw_small -model_path ../models/SMALLBATCH/ -train_from ../models/models/model_step_148000.pt
#python train.py  -task abs -mode train -bert_data_path ../bert_data/random_doc/bert_data -dec_dropout 0.2   -sep_optim true -lr_bert 0.002 -lr_dec 0.2 -save_checkpoint_steps 350 -batch_size 10 -train_steps 157000 -report_every 50 -accum_count 1 -use_bert_emb true -use_interval true -warmup_steps_bert 20000 -warmup_steps_dec 10000 -max_pos 512 -visible_gpus 0 -log_file ../logs/abs_bert_cnndm_freq_save -model_path ../models/SMALLBATCH_RAND/ -train_from ../models/models/model_step_148000.pt
#Fine tune better #2:
#python train.py  -task abs -mode train -bert_data_path ../bert_data/deterministic/bert_data -dec_dropout 0.2  -sep_optim true -lr_bert 0.00002 -lr_dec 0.005 -save_checkpoint_steps 200 -batch_size 25 -train_steps 200000 -report_every 10 -accum_count 1 -use_bert_emb true -use_interval true -warmup_steps_bert 200 -warmup_steps_dec 200 -max_pos 512 -visible_gpus 0 -log_file ../logs/random_freq_save -model_path ../models/FREQUENT_SAVES_DETER_DOC/ -train_from ../../GITHUB/PreSumm/models/model_step_148000.pt

#python train.py -task abs -mode validate -batch_size 50 -test_batch_size 600 -bert_data_path ../bert_data/abs/bert_data -log_file ../logs/EVAL_ALL_RAND -model_path ../models/ENC_ABS_10xDEC_12K_LR02 -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 50 -alpha 0.95 -min_length 5 -result_path ../logs/EVAL_ALL_REASULTS_NEW_RAND

#python train.py -task abs -mode validate -batch_size 25 -test_batch_size 500 -bert_data_path ../bert_data/random_doc/bert_data -log_file ../logs/EVAL_ALL_RAND -model_path ../models/SMALLBATCH_RAND -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 5 -result_path ../logs/EVAL_ALL_RESULTS_NEW_RAND

#python train.py -task abs -mode test -batch_size 50 -test_batch_size 100 -bert_data_path ../bert_data/bert_data -log_file ../logs/test_148700 -model_path ../models/FREQUENT_SAVES -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 5 -result_path ../logs/test_148700 -test_from ../models/FREQUENT_SAVES/model_step_148700.pt -text_src nofile

#python train.py -task abs -mode test -batch_size 50 -test_batch_size 100 -bert_data_path ../bert_data/bert_data -log_file ../logs/test_148700 -model_path ../models/FREQUENT_SAVES -sep_optim true -use_interval true -visible_gpus 0 -max_pos 512 -max_length 200 -alpha 0.95 -min_length 5 -result_path ../logs/test_148700 -test_from ../models/FREQUENT_SAVES/model_step_148000.pt -text_src nofile


#Use: model_step_148700.pt

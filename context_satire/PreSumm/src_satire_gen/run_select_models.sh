#!/bin/bash

#conda activate myenv

#FILES='../../../PreSumm/models/15_200_BATCH_LR02/*' #3750.pt 15K_200_BATCH_LR02
TEMP=1  #'../../../PreSumm/models/15K_200_BATCH/model_step_148500.pt
for f in ../../../PreSumm/models/unfun_ENC00002_DEC02_W500/model_step_340.pt 
#ENC00002_DEC02_500_500_WARMUP/* # ENC*/* #ENC_MORE_DEC_12K_LR02/model_step_2750.pt

#ENC_ABS_100MORE_DEC_12K_LR02/model_step_3250.pt #  

#ENC_MORE_DEC_12K_LR02/model_step_2500.pt
# ../../../PreSumm/models/ENC_12K_LR2/model_step_2000.pt

#CLEAN_12K_LR02_CONT/model_step_153800.pt # models/model_step_148000.pt #FREQUENT_SAVES_300_BATCH/model_step_149* #000.pt

#../../../PreSumm/models/*/* #' #15K_200_BATCH/model_step_148750.pt'
# '../../../PreSumm/models/models/model_step_148700.pt' '../../../PreSumm/models/FREQUENT_SAVES_300_BATCH_model_step_148500.pt' '../../../PreSumm/models/FREQUENT_SAVES_300_BATCH/model_step_148750.pt' '../../../PreSumm/models/FREQUENT_SAVES_FIRST_CONFIG/model_step_148600.pt' '../../../PreSumm/models/FREQUENT_SAVES_FIRST_CONFIG/model_step_148800.pt'

# ../../../PreSumm/models/15K_200_BATCH/model_step_148*

  #LINEAR_TUNE_CONTLR01/* #15K_150_BATCH_LR02/model_step_151000.pt   #$FILES #/model_step_151250.pt"

#"../../../PreSumm/models/15K_200_BATCH/model_step_150250.pt" "../../../PreSumm/models/15K_200_BATCH/model_step_150500.pt" "../../../PreSumm/models/15K_200_BATCH/model_step_151250.pt" "../../../PreSumm/models/15K_200_BATCH/model_step_149800.pt" "../../../PreSumm/models/15K_200_BATCH/model_step_150000.pt" "../../../PreSumm/models/15K_200_BATCH/model_step_152500.pt" "../../../PreSumm/models/15K_200_BATCH/model_step_152000.pt"                                     

#'../../../PreSumm/models/models/model_step_148700.pt' '../../../PreSumm/models/FREQUENT_SAVES_300_BATCH_model_step_148500.pt' '../../../PreSumm/models/FREQUENT_SAVES_300_BATCH/model_step_148750.pt' '../../../PreSumm/models/FREQUENT_SAVES_FIRST_CONFIG/model_step_148600.pt' '../../../PreSumm/models/FREQUENT_SAVES_FIRST_CONFIG/model_step_148800.pt'
do
if test ${f: -3} = '.pt'; then

echo $f

save_file=${f//\//_}
save_file_clean=${save_file//./_}

echo ${f:0:31}

if test ${f:0:31} = '../../../PreSumm/models/ENC_ABS'; then
#python  -W ignore::UserWarning  train.py -task abs -mode test_text -log_file log_select_$TEMP.txt -sep_optim true -use_interval true -visible_gpus 0  -max_pos 512 -max_length 60 -alpha 0.95 -min_length 5 -result_path results_select_temp_$TEMP.txt -test_from $f -text_src news_doc_abs_out_style >> GEN_ALL_OUT_ENC_NEWS/${TEMP}_$save_file
echo "Skipping"

else 

echo "Skipping"
python  -W ignore::UserWarning  train.py -task abs -mode test_text -log_file log_select_$TEMP.txt -sep_optim true -use_interval true -visible_gpus 0  -max_pos 512 -max_length 60 -alpha 0.95 -min_length 5 -result_path results_select_temp_$TEMP.txt -test_from $f -text_src all_test_input_unfun.txt >> POST_SUBMIT_GEN/FIXED${TEMP}_$save_file    #news_doc_out_style.txt #>> GEN_ALL_OUT_ENC_NEWS/${TEMP}_$save_file
#python  -W ignore::UserWarning  train.py -task abs -mode test_text -log_file log_select_$TEMP.txt -sep_optim true -use_interval true -visible_gpus 0  -max_pos 512 -max_length 60 -alpha 0.95 -min_length 5 -result_path results_select_temp_$TEMP.txt -test_from $f -text_src test_generation_input.txt >> GEN_ALL_OUT_ENC_NEWS/${TEMP}_$save_file

 #recent_news_out.txt # example_sum_advers.txt #recent_news_out.txt #>> RECENT_NEWS/${TEMP}_$save_file
#python  -W ignore::UserWarning  train.py -task abs -mode test_text -log_file log_select_$TEMP.txt -sep_optim true -use_interval true -visible_gpus -1 -max_pos 512 -max_length 40 -alpha 0.95 -min_length 5 -result_path results_select_temp_$TEMP.txt -test_from $f -text_src 15K_testing_data_raw_no_fill.txt  >> 15K_without_report_top100/LR02_${TEMP}_$save_file
fi

fi
done

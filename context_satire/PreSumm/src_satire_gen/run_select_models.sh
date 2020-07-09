#!/bin/bash


INPUT='test_generation_input.txt'

for f in ../../PreSumm/models/*
do
if test ${f: -3} = '.pt'; then

echo $f

save_file=${f//\//_}
save_file_clean=${save_file//./_}

echo ${f:0:31}
python  -W ignore::UserWarning  train.py -task abs -mode test_text -log_file log.txt -sep_optim true -use_interval true -visible_gpus 0  -max_pos 512 -max_length 60 -alpha 0.95 -min_length 5 -result_path results_select_$f -test_from $f -text_src $INPUT 

fi
fi
done

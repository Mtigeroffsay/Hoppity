#!/bin/bash

TARGET_MODEL=./saves/cwe_small_JD_save/epoch-91.ckpt
DATA_ROOT=./gtrans/dataset_processed/cwe_small_JD_cooked
DATA_NAME=cwe_small_JD
SAVE_DIR=./saves/cwe_small_JD_save/test



#nohup python ./gtrans/eval/eval_text.py -target_model $TARGET_MODEL -data_root $DATA_ROOT -data_name $DATA_NAME -save_dir $SAVE_DIR -iters_per_val 100 -beam_size 3 -topk 3 -gnn_type 's2v_multi' -max_lv 4 -gpu 0 -resampling True -comp_method 'mlp' -bug_type True -loc_acc True -val_acc True -op_acc True -type_acc True -output_all True -batch_size 1 &> test_log_1219.out &

python ./gtrans/eval/eval_text.py -target_model $TARGET_MODEL -data_root $DATA_ROOT -data_name $DATA_NAME -save_dir $SAVE_DIR -iters_per_val 100 -beam_size 3 -topk 3 -gnn_type 's2v_multi' -max_lv 4 -gpu 1 -resampling True -comp_method 'mlp' -bug_type True -loc_acc True -val_acc True -op_acc True -type_acc True -output_all True -batch_size 1


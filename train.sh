#!/bin/bash

DATA_PATH=./gtrans/dataset_processed/cwe_small_JD_cooked
DATA_NAME=cwe_small_JD
SAVE_PATH=./saves/cwe_small_JD_save

 
python ./gtrans/training/main_gtrans.py \
  -data_root  $DATA_PATH \
  -data_name  $DATA_NAME \
  -save_dir   $SAVE_PATH \
  -gnn_type  "s2v_multi" \
  -max_lv  4 \
  -resampling  True \
  -comp_method  "mlp" \
  -batch_size  2 \
  -max_ast_nodes 3000\
  -num_epochs 100\
  -output_all True \
  -use_colab True \
  $@
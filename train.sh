#!/bin/bash

DATA_PATH=./gtrans/dataset_processed/cwe_full_processed
DATA_NAME=cwe_full
SAVE_PATH=./saves/cwe_full

nohup python ./gtrans/training/main_gtrans.py \
  -data_root  $DATA_PATH \
  -data_name  $DATA_NAME \
  -save_dir   $SAVE_PATH \
  -gnn_type  "s2v_multi" \
  -max_lv  4 \
  -resampling  True \
  -comp_method  "mlp" \
  -batch_size  2 \
  -max_ast_nodes 3000\
  -num_epochs 300\
  -output_all True \
  -gpu 0 \
  $@ &> cwe_full_1231.out &

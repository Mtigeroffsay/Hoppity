data_name=cwe_full_JD
cooked_root=./gtrans/dataset_processed/cwe_full_JD_cooked
save_dir=./saves/cwe_full_JD_save/statistics/
target_model=./saves/cwe_full_JD_save/epoch-115.ckpt

#data_name=cwe_small_JD
#cooked_root=./gtrans/dataset_processed/cwe_small_JD_cooked
#save_dir=./saves/cwe_small_JD_save/statistics/
#target_model=./saves/cwe_small_JD_save/epoch-99.ckpt


##export CUDA_VISIBLE_DEVICES=0

python ./gtrans/eval/eval_text.py \
	-target_model $target_model \
	-data_root $cooked_root \
	-data_name $data_name \
	-save_dir $save_dir \
	-iters_per_val 100 \
	-beam_size 1 \
	-batch_size 1 \
	-topk 1 \
	-gnn_type 's2v_multi' \
	-max_lv 4 \
	-max_modify_steps 100 \
	-gpu 0 \
	-resampling True \
	-comp_method "mlp" \
	-bug_type True \
	-loc_acc True \
	-val_acc True \
	-output_all True \
	$@

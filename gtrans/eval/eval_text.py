import os
import numpy as np
from time import time
import torch
from tqdm import tqdm
from gtrans.eval.utils import ast_acc_cnt, setup_dicts, loc_acc_cnt, val_acc_cnt, type_acc_cnt, op_acc_cnt, get_top_k, get_val
from gtrans.data_process.utils import get_bug_prefix
from gtrans.common.configs import cmd_args
from gtrans.common.dataset import Dataset, GraphEditCmd
from gtrans.model.gtrans_model import GraphTrans
from gtrans.common.consts import DEVICE
from gtrans.common.consts import OP_REPLACE_VAL, OP_ADD_NODE, OP_REPLACE_TYPE, OP_DEL_NODE, OP_NONE
import gc

import pdb

const_val_vocab = np.load(os.path.join(cmd_args.data_root, "vocab_" + cmd_args.vocab_type + ".npy"), allow_pickle=True).item()
#pdb.set_trace()
Dataset.set_value_vocab(const_val_vocab)
Dataset.add_value2vocab(None)
Dataset.add_value2vocab("UNKNOWN")

dataset = Dataset(cmd_args.data_root, cmd_args.gnn_type)
dataset.load_partition()

phase = "test"
import sys
sys.setrecursionlimit(50000)
torch.set_num_threads(12)

def printTree(node,tokenList):
    if node.is_leaf and type(node.value)==str:
        tokenList.append(node.value)
    else:
        for n in node.children:
            printTree(n,tokenList)

def printTree2(node):
    if node.is_leaf and type(node.value)==str:
        print(node.value,end=' ')
    else:
        for n in node.children:
            printTree2(n)


def sample_gen(s_list):
    yield s_list


def get_gt_edit_len(sample_list):
    return [len(sample.g_edits) for sample in sample_list]


#either input a list of inputs or just generate some from the test set 
if not cmd_args.sample_list:
    train_gen = dataset.data_gen(cmd_args.batch_size, phase='train', infinite=False)
    test_gen = dataset.data_gen(cmd_args.batch_size, phase='test', infinite=False)
else:
    new_sample_list = []
    for sample in cmd_args.sample_list:
        new_sample_list.append(dataset.get_sample(sample)) 

    val_gen = sample_gen(new_sample_list)

model = GraphTrans(cmd_args).to(DEVICE)
print("loading", cmd_args.target_model)

if cmd_args.rand:
    model.set_rand_flag(True)

model.load_state_dict(torch.load(cmd_args.target_model))
model.eval()


###### test statistics
'''
edit_len_list = []
edit_len_list_gt = []


for sample_list in tqdm(test_gen):
    _, _, stop_steps_batch, _ = model(sample_list, phase='test', beam_size=cmd_args.beam_size, pred_gt=False, op_given=cmd_args.op_given, loc_given=cmd_args.loc_given)
    edit_len_list += stop_steps_batch
    del stop_steps_batch
    el_batch_gt = get_gt_edit_len(sample_list)
    edit_len_list_gt += el_batch_gt
    del el_batch_gt


fres=open(cmd_args.save_dir+"/test_edit_lengths.txt","w",encoding="ascii",buffering=1)
fres.write("test set statistics:\n")
fres.write(f"total number of files: %d\n" %len(edit_len_list_gt))
fres.write("predicted edit sequences:\n")
fres.write(f"mean edit length: %.2f\n" %np.mean(edit_len_list))
fres.write(f"edit length std: %.2f\n" %np.std(edit_len_list))
fres.write("ground truth edit sequences:\n")
fres.write(f"mean edit length: %.2f\n" %np.mean(edit_len_list_gt))
fres.write(f"edit length std: %.2f\n" %np.std(edit_len_list_gt))
fres.close()


print("test set evaluation complete!")

print("edit length (prediction & ground truth):")
print(edit_len_list)
print(edit_len_list_gt)
'''
###### training statistics

edit_len_list = []
edit_len_list_gt = []

for sample_list in tqdm(train_gen):
    #ll, new_asts, stop_steps_batch, reg_batch_raw = model(sample_list, phase='test', beam_size=cmd_args.beam_size, pred_gt=False, op_given=cmd_args.op_given, loc_given=cmd_args.loc_given)
    #edit_len_list += stop_steps_batch
    el_batch_gt = get_gt_edit_len(sample_list)
    edit_len_list_gt += el_batch_gt
    del el_batch_gt

'''
fres=open(cmd_args.save_dir+"/train_edit_lengths.txt","w",encoding="ascii",buffering=1)
fres.write("training set statistics:\n")
fres.write(f"total number of files: %d\n" %len(edit_len_list_gt))
#fres.write("predicted edit sequences:\n")
#fres.write(f"mean edit length: %.2f\n" %np.mean(edit_len_list))
#fres.write(f"mean edit length: %.2f\n" %np.std(edit_len_list))
fres.write("ground truth edit sequences:\n")
fres.write(f"mean edit length: %.2f\n" %np.mean(edit_len_list_gt))
fres.write(f"edit length std: %.2f\n" %np.std(edit_len_list_gt))
fres.close()
'''
print(np.min(edit_len_list_gt))
print(np.max(edit_len_list_gt))
print(np.std(edit_len_list_gt))
print("training set evaluation complete!")
'''
print("edit length (ground truth):")
print(edit_len_list_gt)
'''


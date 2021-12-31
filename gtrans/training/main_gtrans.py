from __future__ import print_function
from __future__ import absolute_import
from __future__ import division


import random
import torch
import os
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from gtrans.common.configs import cmd_args
from gtrans.common.consts import DEVICE
from gtrans.common.dataset import Dataset
from gtrans.model.gtrans_model import GraphTrans
from gtrans.common.code_graph import tree_equal
import gc

import pdb

import pickle

def ast_acc_cnt(pred_asts, true_asts, contents):
    count = 0
    acc = 0
    assert len(pred_asts) == len(true_asts)
    for x_list, y, c in zip(pred_asts, true_asts, contents):
        x_list = x_list[:cmd_args.topk]
        count += 1
        for x in x_list:
            if tree_equal(x, y, c):
                acc += 1
                break
    return acc


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.set_num_threads(1)
    torch.manual_seed(cmd_args.seed)
    torch.autograd.set_detect_anomaly(True)
    
    
    vocab_name = 'vocab_%s.npy' % cmd_args.vocab_type
    print('loading value vocab from', vocab_name)
    const_val_vocab = np.load(os.path.join(cmd_args.data_root, vocab_name), allow_pickle=True).item()
    Dataset.set_value_vocab(const_val_vocab)
    Dataset.add_value2vocab(None)
    Dataset.add_value2vocab("UNKNOWN")
    print('global value table size', Dataset.num_const_values())

    dataset = Dataset(cmd_args.data_root, cmd_args.gnn_type, 
                      data_in_mem=cmd_args.data_in_mem,
                      resampling=cmd_args.resampling)

    dataset.load_partition()
    train_gen = dataset.data_gen(cmd_args.batch_size, phase='train', infinite=True)

    best_test_loss = None
    model = GraphTrans(cmd_args).to(DEVICE)
    if cmd_args.init_model_dump is not None:
        model.load_state_dict(torch.load(cmd_args.init_model_dump))
        stats_file = os.path.join(os.path.dirname(cmd_args.init_model_dump), 'best_val.stats')
        if os.path.isfile(stats_file):
            with open(stats_file, 'r') as f:
                best_test_loss = float(f.readline().strip())
                
    ### load the model
    start_epoch = 0
    #model.load_state_dict(torch.load("./cwe_full_save/epoch-144.ckpt"))
    
    optimizer = optim.Adam(model.parameters(), lr=cmd_args.learning_rate)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1, last_epoch=-1)

    train_epoch_losses = []

    for epoch in range(cmd_args.num_epochs):
        total_loss = 0 
        total_itrs = 0
        model.train()
        pbar = tqdm(range(cmd_args.iters_per_val))
        for it in pbar:
            try:
                sample_list = next(train_gen)
                optimizer.zero_grad()
                
                #pdb.set_trace()

                #ll, new_asts = model(sample_list, phase='train', pred_gt=True)
                #ll, new_asts, stop_steps_batch, reg_batch_raw = model(sample_list, phase='train', pred_gt=True)
                
                ll, new_asts, _, _ = model(sample_list, phase='train', pred_gt=True)

                #pdb.set_trace()
                           

                loss = -torch.mean(ll)
                loss.backward()
                if cmd_args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cmd_args.grad_clip)

                optimizer.step()
                epoch_num = (epoch + (it + 1) /cmd_args.iters_per_val)
                total_loss += loss.item()
                total_itrs += 1
                pbar.set_description('epoch %.2f, loss: %.4f, avg loss: %.4f' % (epoch_num, loss.item(), total_loss / total_itrs))
            except:
                print('OOM')
                try:
                    del ll
                except:
                    pass
                try:
                    del new_asts
                except:
                    pass
                try:
                    del loss
                except:
                    pass
                torch.cuda.empty_cache()
                gc.collect()
        epoch_bavg_loss = total_loss/total_itrs
        train_epoch_losses.append(epoch_bavg_loss)
        del epoch_bavg_loss


        if epoch % 1 == 0:
            torch.save(model.state_dict(), os.path.join(cmd_args.save_dir, 'epoch-%d.ckpt' % (epoch)))
            pickle.dump(train_epoch_losses, open(os.path.join(cmd_args.save_dir, 'train_epoch_losses.pkl'), "wb"))

        cur_lr = scheduler.get_lr()
        if cur_lr[-1] > cmd_args.min_lr:
            scheduler.step()

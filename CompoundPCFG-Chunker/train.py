#!/usr/bin/env python3
import sys
import os

import argparse
import json
import random
import shutil
import copy
import pickle
from tqdm import tqdm 

import torch
from torch import cuda
import numpy as np
import time
import logging
from data import Dataset
from utils import *
from models import CompPCFG
from torch.nn.init import xavier_uniform_
from subprocess import run, PIPE

parser = argparse.ArgumentParser()

# Data path options
parser.add_argument('--train_file', default='data/ptb-train.pkl')
parser.add_argument('--val_file', default='data/ptb-val.pkl')
parser.add_argument('--save_path', default='german/', help='where to save the model')

# Model options
# Generative model parameters
parser.add_argument('--z_dim', default=64, type=int, help='latent dimension')
parser.add_argument('--t_states', default=60, type=int, help='number of preterminal states')
parser.add_argument('--nt_states', default=30, type=int, help='number of nonterminal states')
parser.add_argument('--state_dim', default=256, type=int, help='symbol embedding dimension')
# Inference network parameters
parser.add_argument('--h_dim', default=512, type=int, help='hidden dim for variational LSTM')
parser.add_argument('--w_dim', default=512, type=int, help='embedding dim for variational LSTM')
# Optimization options
parser.add_argument('--num_epochs', default=10, type=int, help='number of training epochs')
parser.add_argument('--lr', default=0.001, type=float, help='starting learning rate')
parser.add_argument('--max_grad_norm', default=3, type=float, help='gradient clipping parameter')
parser.add_argument('--max_length', default=30, type=float, help='max sentence length cutoff start')
parser.add_argument('--len_incr', default=1, type=int, help='increment max length each epoch')
parser.add_argument('--final_max_length', default=40, type=int, help='final max length cutoff')
parser.add_argument('--beta1', default=0.75, type=float, help='beta1 for adam')
parser.add_argument('--beta2', default=0.999, type=float, help='beta2 for adam')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')
parser.add_argument('--seed', default=3435, type=int, help='random seed')
parser.add_argument('--print_every', type=int, default=1000, help='print stats after N batches')

def build_vocab(t_data, v_data, test_data):
  word_to_ix = {}
  net = []
  for a in t_data:
    net.append(a)
  for c in v_data:
    net.append(c)
  for e in test_data:
    #print(e)
    net.append(e)
  
  for sent in net:
    for word in sent:
      if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)
  
  ix_to_word = {v: k for k, v in word_to_ix.items()}
  
  return word_to_ix, ix_to_word

def valid_conll_eval(fname):

  with open(fname, 'r') as file:
    data = file.read()

  pipe = run(["perl", "eval_conll2000_updated.pl"], stdout=PIPE, input=data, encoding='ascii')
  output = pipe.stdout
  tag_acc = float(output.split()[0])
  phrase_f1 = float(output.split()[1])
  return phrase_f1

def main(args):
  train_data = pickle.load(open("data/conll/data_train_tokens.pkl", "rb"))
  val_data = pickle.load(open("data/conll/data_val_tokens.pkl", "rb"))
  test_data = pickle.load(open("data/conll/data_test_tokens.pkl", "rb"))
  #BI_test_gt = pickle.load(open("../../german_small_updated/german_test2_tag.pkl", "rb"))
  BI_val_gt = pickle.load(open("data/conll/data_val_tags.pkl", "rb"))
  
  # train_data = train_data[:200]
  # val_data = val_data[:200]
  # BI_val_gt = BI_val_gt[:200]

  BI_val_gt = np.concatenate([np.array(g) for g in BI_val_gt])   #original code
  word_to_ix, ix_to_word = build_vocab(train_data, test_data, val_data)

  np.random.seed(args.seed)
  torch.manual_seed(args.seed)
  # train_data = Dataset(args.train_file)
  # val_data = Dataset(args.val_file)  
  # train_sents = train_data.batch_size.sum()
  # vocab_size = int(train_data.vocab_size)    
  # max_len = max(val_data.sents.size(1), train_data.sents.size(1))
  
  ###### new ########
  train_sents = len(train_data)
  vocab_size = len(word_to_ix)    
  max_len = 20
  
  print('Vocab size: %d, Max Sent Len: %d' % (vocab_size, max_len))
  print('Save Path', args.save_path)
  
  # cuda.set_device(args.gpu)
  device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  model = CompPCFG(vocab = vocab_size,
                   state_dim = args.state_dim,
                   t_states = args.t_states,
                   nt_states = args.nt_states,
                   h_dim = args.h_dim,
                   w_dim = args.w_dim,
                   z_dim = args.z_dim)
  for name, param in model.named_parameters():    
    if param.dim() > 1:
      xavier_uniform_(param)
  #print("model architecture")
  #print(model)
  model.train()
  # model.cuda()
  model.to(device)
  optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas = (args.beta1, args.beta2))
  best_val_ppl = 1e5
  best_val_f1 = 0
  epoch = 0
  best_fscore = 0.

  while epoch < args.num_epochs:
    start_time = time.time()
    epoch += 1  
    print('Starting epoch %d' % epoch)
 
    for i in tqdm(np.random.permutation(len(train_data))):
      # sents, length, batch_size, _, gold_spans, gold_binary_trees, _ = train_data[i]  
      # print("ptb sents", sents)
      # exit()

      sents = torch.tensor([[word_to_ix[word] for word in train_data[i]]])
      length = len(sents[0])
      # print("conll sentences")
      # print(sents)
      # print("length")
      # print(length)
    

      # if length > args.max_length or length == 1: #length filter based on curriculum 
      #   continue
      # sents = sents.cuda()
      sents = sents.to(device)
      optimizer.zero_grad()
      nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True)      
      (nll+kl).mean().backward()
  
      torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)      
      optimizer.step()
     
    print('--------------------------------')
    print('Checking validation perf...') 
    pred_path_val_out = args.save_path + 'validation/' + str(epoch) + '.out'   
    eval(val_data, model, BI_val_gt, pred_path_val_out, word_to_ix)
    fscore = valid_conll_eval(pred_path_val_out)
    print('--------------------------------')
    if fscore > best_fscore:
      print("fscore", fscore)
      best_fscore = fscore
      checkpoint = {
        'args': args.__dict__,
        'model': model.cpu(),
        'word2idx': word_to_ix
      }
      print('Saving checkpoint to %s' % args.save_path)
      torch.save(checkpoint, args.save_path+'compound_pcfg.pt')
      # model.cuda()
      model.to(device)

def eval(data, model, BI_gt, pred_path, word_to_ix):
  model.eval()
  
  with torch.no_grad():

    with open(pred_path, 'a') as fp:

      pred_list = []
      BI_pred = []
      s_id=0
      checker = 0
      for i in tqdm(range(len(data))):
        #sents, length, batch_size, _, gold_spans, gold_binary_trees, other_data = data[i] 
        sents = torch.tensor([[word_to_ix[word] for word in data[i]]])
        length = len(sents[0])

        # if length == 1:
        #   continue
        # sents = sents.cuda()
        sents = setns.to(device)
        
        nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True)
        binary_matrix = binary_matrix[0].detach().cpu().numpy() 

        pred = np.zeros(binary_matrix.shape[0]).astype(object)
        for i in range(binary_matrix.shape[0]):
          word_vec = binary_matrix[i]
          for j in range(len(word_vec)):
            chain = 1
            if j <= i:
              continue
            else:
              if chain == 1 and word_vec[j] == 1.:
                if j == i + 1:
                  pred[i] = 'B'                 
                pred[j] = 'I'
                chain = 1
              else:
                chain = 0
                continue
        #print(pred)
        pred[pred == 0.] = 'B'
        #print(pred_list)
        #print("###")
        pred_list.append(pred)
      
      print("no. of sents processed:", len(pred_list))

      for l in range(len(pred_list)):
        for m in range(len(pred_list[l])):
          if pred_list[l][m] != 0:
            BI_pred.append(pred_list[l][m])

      print("Num of elements in pred: ", len(BI_pred))
      print("Num of elements in gt: ", len(BI_gt))
      for p in range(len(BI_pred)):
        if BI_gt[p] == 'b':
          BI_gt[p] = 'B'
        elif BI_gt[p] == 'i':
          BI_gt[p] = 'I'
        fp.write("x "+"y "+str(BI_gt[p])+" "+str(BI_pred[p]))
        fp.write("\n")

  model.train()  
  return 0

if __name__ == '__main__':
  args = parser.parse_args()
  main(args)

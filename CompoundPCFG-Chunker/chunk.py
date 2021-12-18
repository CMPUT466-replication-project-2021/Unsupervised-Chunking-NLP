#!/usr/bin/env python3
import sys
import os
import torch
from torch import cuda
import torch.nn as nn
import numpy as np
import time
import re
import pickle
import argparse
import json
import random
import shutil
from tqdm import tqdm
import copy
from utils import *

parser = argparse.ArgumentParser()


parser.add_argument('--data_file', default='train/conll_train.txt')
parser.add_argument('--tag_file', default='train/data_train_tags.pkl')
parser.add_argument('--model_file', default='data/trained-fmodels/compound-pcfg.pt')
parser.add_argument('--out_file')
# Inference options
parser.add_argument('--use_mean', default=1, type=int, help='use mean from q if = 1')
parser.add_argument('--gpu', default=0, type=int, help='which gpu to use')


device = torch.device('cpu')
print(device)

def clean_number(w):    
	new_w = re.sub('[0-9]{1,}([,.]?[0-9]*)*', 'N', w)
	return new_w
  
def main(args):
	
	data_tags_gt = pickle.load(open(args.tag_file, "rb"))
	
	BI_gt = np.concatenate([np.array(g) for g in data_tags_gt])   #original code
	#BI_gt = data_tags_gt
	target_names = ['B', 'I']  

	print('loading model from ' + args.model_file)
	checkpoint = torch.load(args.model_file)
	model = checkpoint['model'].to(device)
	#cuda.set_device(args.gpu)
	model.eval()
	#model.cuda()
	word2idx = checkpoint['word2idx']
	#print(word2idx['Freizeit'])

	pred_out = open(args.out_file, "w")
	total_time = 0.
	with torch.no_grad():
		pred_path = args.out_file
		fc = 0	

		with open(pred_path, 'a') as fp:

			pred_list = []
			BI_pred = []
			s_id=0
			checker = 0
			for sent_orig in tqdm(open(args.data_file, "r")):
				#print(sent_orig)
				#print(checker)
				checker+=1
				sent = sent_orig.strip().split()
				#sent = [clean_number(w) for w in sent]
				length = len(sent)

				if length == 1:
					continue
				
				# for word in sent:
				# 	if word not in word2idx: 
				# 		#print(checker, word)
						
				#sent_idx = [word2idx[w] if w in word2idx else 0 for w in sent] 
				sent_idx = [word2idx[w] if w in word2idx else word2idx["<unk>"] for w in sent]

				#print(sent_idx) 
				sents = torch.from_numpy(np.array(sent_idx)).unsqueeze(0)				
				start_time = time.time()
				nll, kl, binary_matrix, argmax_spans = model(sents, argmax=True, use_mean=(args.use_mean==1))
				binary_matrix = binary_matrix[0].detach().cpu().numpy() 
				end_time = time.time()
				process_time = end_time - start_time
				total_time = total_time + process_time

				#################### maximal left branching chunks ##############################
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

				pred[pred == 0.] = 'B'
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
				fc+=1
				fp.write("x "+"y "+str(BI_gt[p])+" "+str(BI_pred[p]))
				fp.write("\n")
	print("total_time", total_time)

if __name__ == '__main__':
	args = parser.parse_args()
	main(args)

import numpy as np
import pickle
import collections


x = pickle.load(open("train/data_train_tokens.pkl", "rb"))
y = pickle.load(open("val/data_val_tokens.pkl", "rb"))
z = pickle.load(open("test/data_test_tokens.pkl", "rb"))


pred_path = 'train/conll_train.txt' # options: 'train/conll_train.txt', 'val/conll_val.txt', 'test/conll_test.txt'

fc = 0
with open(pred_path, 'a') as fp:
	#for data_kind in [x,y,z]:
	for data_kind in [x]: #options: [x], [y], [z]

		for p in range(len(data_kind)):
			fc+=1
			#if p % 50 == 0:	
				# data_kind[p].insert(0, 'SOS')
				# data_kind[p].append('EOS')
				# data_kind[p].append('PAD')
			
			sent = ' '.join(word for word in data_kind[p])
			fp.write(sent)
			fp.write("\n")
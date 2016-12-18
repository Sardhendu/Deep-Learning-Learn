#####
# Name : Sardhendu Mishra
import os
import sys

import numpy as np
import scipy.io as sio
from six.moves import cPickle as pickle


data_load_from  = '/Users/sam/All-Program/App-DataSet/Deployment-Code/Word-Search-NNets/Mat_files/'

pickle_compressed = '/Users/sam/All-Program/App-DataSet/Deployment-Code/Word-Search-NNets/Pickle_files/dataset_complete.p'

# The indices of the vocab ranges from 1 to 250, so we subtract 1 from each cell in ndarray and convert the range to 0,250 
def load_data_from ():
	train_input = sio.loadmat(data_load_from+'train_input.mat')
	train_input = train_input['train_input']-1
	# print np.amin(train_input)
	# print np.amax(train_input)

	train_target = sio.loadmat(data_load_from+'train_target.mat')
	train_target = train_target['train_target']-1
	# print np.amin(train_target)
	# print np.amax(train_target)

	valid_input = sio.loadmat(data_load_from+'valid_input.mat')
	valid_input = valid_input['valid_input']-1
	# print np.amin(valid_input)
	# print np.amax(valid_input)

	valid_target = sio.loadmat(data_load_from+'valid_target.mat')
	valid_target = valid_target['valid_target']-1
	# print np.amin(valid_target)
	# print np.amax(valid_target)

	test_input = sio.loadmat(data_load_from+'test_input.mat')
	test_input = test_input['test_input']-1
	# print np.amin(test_input)
	# print np.amax(test_input)

	test_target = sio.loadmat(data_load_from+'test_target.mat')
	test_target = test_target['test_target']-1
	# print np.amin(test_target)
	# print np.amax(test_target)

	vocab = sio.loadmat(data_load_from+'vocab.mat')
	vocab = vocab['vocab']

	return train_input, train_target, valid_input, valid_target, test_input, test_target, vocab


def load_data_to (training_dataset, training_labels, crossvalid_dataset, crossvalid_labels, test_dataset, test_labels, vocab):
	try:
	    f = open(pickle_compressed, 'wb')
	    dataset_complete = {
	        'training_dataset': training_dataset,
	        'training_labels': training_labels,
	        'crossvalid_dataset': crossvalid_dataset,
	        'crossvalid_labels': crossvalid_labels,
	        'test_dataset': test_dataset,
	        'test_labels': test_labels,
	        'vocab': vocab,
	    }
	    pickle.dump(dataset_complete, f, pickle.HIGHEST_PROTOCOL)
	    f.close()
	except Exception as e:
	    print('Unable to save data to', pickle_compressed, ':', e)
	    raise
	    
	statinfo = os.stat(pickle_compressed)
	print('Compressed pickle size:', statinfo.st_size)

train_input, train_target, valid_input, valid_target, test_input, test_target, vocab = load_data_from()

load_data_to(train_input, train_target, valid_input, valid_target, test_input, test_target, vocab)


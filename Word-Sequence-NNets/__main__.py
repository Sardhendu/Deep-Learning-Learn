#####
# Name : Sardhendu Mishra

from __future__ import division

import numpy as np
import pandas as pd
from frwd_propagate import fprop
from six.moves import cPickle as pickle

pickle_compressed = '/Users/sam/All-Program/App-DataSet/Deployment-Code/Word-Search-NNets/Pickle_files/dataset_complete.p'


######################### Load Data  ######################################
def get_data(pickle_compressed_path): 
	with open(pickle_compressed_path, 'rb') as f:
	    dataset = pickle.load(f)
	    training_dataset = (dataset['training_dataset'])
	    training_labels = (dataset['training_labels'])
	    crossvalid_dataset = (dataset['crossvalid_dataset'])
	    crossvalid_labels = (dataset['crossvalid_labels'])
	    test_dataset = (dataset['test_dataset'])
	    test_labels = (dataset['test_labels'])
	    vocab = (dataset['vocab'])
	return training_dataset, training_labels, crossvalid_dataset, crossvalid_labels, test_dataset, test_labels, vocab

training_dataset, training_labels, crossvalid_dataset, crossvalid_labels, test_dataset, test_labels, vocab = get_data(pickle_compressed)
print training_dataset.shape
print training_labels.shape
print crossvalid_dataset.shape
print crossvalid_labels.shape
print test_dataset.shape
print test_labels.shape
print vocab.shape
######################### Load Data  ######################################



######################  Set Hyperparameters  ##############################
no_words_ngram, batchsize, no_batches = training_dataset.shape  # Mini-batch size.
vocab_size = vocab.shape[1];
learning_rate = 0.1  # Learning rate; default = 0.1.
momentum = 0.9  # Momentum; default = 0.9.
no_embed_unit = 50  # Dimensionality of embedding space; default = 50.
no_hidden_unit = 200  # Number of units in hidden layer; default = 200.
init_wt = 0.01  # Standard deviation of the normal distribution, which is sampled to get the initial weights; default = 0.01

# VARIABLES FOR TRACKING TRAINING PROGRESS.
show_training_CE_after = 100;
show_validation_CE_after = 1000;

# INITIALIZE WEIGHTS AND BIASES
word_to_embed_wghts = init_wt * np.random.rand(vocab_size, no_embed_unit);  # 250x50
embed_to_hid_wgths = init_wt * np.random.rand(no_words_ngram * no_embed_unit, no_hidden_unit)
hid_to_outpt_wghts = init_wt * np.random.rand(no_hidden_unit, vocab_size);  # 200x250 
hid_bias = np.zeros([no_hidden_unit, 1])
outpt_bias = np.zeros([vocab_size, 1])
count = 0;
noise = np.exp(-30);
######################  Set Hyperparameters  ##############################

def one_hot_coding(labels, no_of_labels):
	# Here we create a matrix with vectors where we have 1 only in the position where the index is ON, 
	"""
	# For example if our vocab list is of 5 words v1,v2,v3,v4,v5 and the actual target output for three tranig set is y1=v2, y2=v4, y3=v5 then we need a one-hot array matrix,
	        v1  v2  v3  v4  v5
		y1  0   1   0   0   0
        y2  0   0   0   1   0
        y3  0   0   0   0   1
	"""
	labels_one_hot = (np.arange(no_of_labels) == labels[:,None]).astype(np.float32)
	return labels_one_hot

def error_derivative_and_CE(train_target_mini_batch, output_layer_state, vocab_size, batch_size, ):  # Cross Entropy
	# Convert the labels into one-hot encoded arrays:
	target_one_hot_coding = one_hot_coding(train_target_mini_batch, vocab_size)
    # Compute Error Derivatives using Cross Entropy
	

	error_deriv = output_layer_state - np.transpose(target_one_hot_coding)

	# Compute the cross Entropy, the less the cross entropy the better the model is. In normal sense entropy measures the unpredictability of a information content. So when the entropy is more it means the output os unpredictable, and when the entropy is less it means there's less surprise which inturns means the hypothesis and actual output are margnally closer. We also average out the total entropy.
	print batch_size
	cross_entropy = ((-1) * np.sum(np.transpose(target_one_hot_coding) * np.log(output_layer_state + noise)))/batch_size
	print cross_entropy
	print error_deriv.shape

	return error_deriv, cross_entropy
	

#def bprop():


# array([[ 0.,  1.,  0.,  0.],
#        [ 1.,  0.,  0.,  0.],
#        [ 0.,  0.,  0.,  1.]])

for i in range(0,no_batches):
	train_input_mini_batch = np.transpose(training_dataset[:,:,i])
	train_target_mini_batch = training_labels[:,:,i]
	print train_input_mini_batch.shape
	print train_target_mini_batch
	# print word_to_embed_wghts.shape
	# print embed_to_hid_wgths.shape
	# print hid_to_outpt_wghts.shape

	# Perform the Forward Propagation
	ebmed_layer_state, hid_layer_state, output_layer_state = fprop(train_input_mini_batch,
			word_to_embed_wghts,
			embed_to_hid_wgths,
			hid_bias,
			hid_to_outpt_wghts,
			outpt_bias)

	error_derivative_and_CE(train_target_mini_batch[0], output_layer_state, vocab_size, train_input_mini_batch.shape[0])
	
	break





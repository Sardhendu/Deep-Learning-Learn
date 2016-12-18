#####
# Name : Sardhendu Mishra
from __future__ import division

import numpy as np


def fprop(input_batch, word_to_embed_wghts, embed_to_hid_wgths, hid_bias, hid_to_outpt_wghts, outpt_bias):
	print 'Function fprop'
	batch_size, no_words_ngram  = input_batch.shape
	vocab_size, no_embed_unit = word_to_embed_wghts.shape

	# The state of the Embedding layer (Hidden layer1)
	ebmed_layer_state = np.reshape(np.transpose(word_to_embed_wghts[np.reshape(input_batch,(1,no_words_ngram*batch_size))[0]]), (no_words_ngram*no_embed_unit, batch_size), order = 'F')
	print 'ebmed_layer_state.shape ', ebmed_layer_state.shape

	# The state of the Hidden2 layer 
	inputs_to_hid = np.dot(np.transpose(embed_to_hid_wgths), ebmed_layer_state) + np.tile(hid_bias, (1,batch_size))
	hid_layer_state = 1 / (1 + np.exp(-inputs_to_hid));   # Using logistic
	print 'hid_layer_state.shape ', hid_layer_state.shape

	# The state of the Output Layer
	inputs_to_softmax = np.dot(np.transpose(hid_to_outpt_wghts), hid_layer_state) + np.tile(outpt_bias, (1,batch_size))
	print 'inputs_to_softmax.shape ', inputs_to_softmax.shape
	inputs_to_softmax = inputs_to_softmax - np.tile(np.max(inputs_to_softmax, axis=0), (vocab_size, 1))
	output_layer_state = np.exp(inputs_to_softmax)
	output_layer_state = output_layer_state / np.tile(np.sum(output_layer_state, axis=0), (vocab_size, 1))
	print 'output_layer_state.shape ', output_layer_state.shape

	# All the values corresponding to any column in the outpu_layer_state would sum to zero.

	return ebmed_layer_state, hid_layer_state, output_layer_state
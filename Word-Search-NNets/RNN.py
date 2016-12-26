import nltk
import itertools
import numpy as np
import csv
import sys  
import pickle

cleaned_dataset_path = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/reddit-comments-2015-08.p'



with open(cleaned_dataset_path, 'rb') as f:
    data = pickle.load(f)
    X_train = (data['X_train'])
    y_train = (data['y_train'])
    
# Assign instance variables
word_dim = vocabulary_size
hidden_dim = 100
bptt_truncate = 4
# Randomly initialize the network parameters
U = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (hidden_dim, hidden_dim))

        

print (-np.sqrt(1./word_dim))
print ('')
print (U.shape)
print (U)
print (V.shape)
print (V)
print (W.shape)
print (W)
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import urllib.request
from six.moves import cPickle as pickle
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader
from gensim import corpora


from model import  dynamic_RNN_model


class Train():
    def __init__(self):
        self.num_hid_units = 3,
        self.momentum = 0.9,
        self.learning_rate = 0.5
        self.train_batch_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/training_batch/'
        dictionary_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/dictionary.txt'
        self.vocab_size = len(corpora.Dictionary.load_from_text(dictionary_dir))

    def accuracy(self, predictions, labels, labels_one_hot = None):
        # The input labels are a One-Hot Vector
        if labels_one_hot:
            return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                  / predictions.shape[0])
        else:
            return (100.0 * np.sum(np.argmax(predictions, 1) == np.reshape(labels, [-1]))
                  / predictions.shape[0])
        

    def train_network(self, graph_dict):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            
            aa = np.array([[1,2,3,4,5], [1,3,4,0,5], [2,3,4,5,6]])  # , [2,3,4,5,6], [2,1,5,4,3]
            bb = np.array([[2,3,4,5,2], [3,4,0,5,2], [3,4,5,6,2]])  # , [3,4,5,6,2], [1,5,4,3,2]
            cc = np.array([[1,4,2,2,1,3], [1,4,3,3,0,0]])
            dd = np.array([[4,2,2,5,3,2], [4,3,0,5,3,0]])
            epochs = 1
            for epoch in np.arange(epochs):
                new_hid_layer_state = None
                print (graph_dict)

                for no in np.arange(10):#[[aa,bb],[cc,dd]]:#np.arange(2):
                    with open(self.train_batch_dir+'batch'+str(no)+'.pickle', 'rb') as f:
                        dataset = pickle.load(f)
                        
                        batch_train_dataset = dataset['batch_train_dataset']
                        batch_train_labels = dataset['batch_train_labels']

                        if not new_hid_layer_state: 
                            feed_dict= {graph_dict['x']: batch_train_dataset, 
                                        graph_dict['y']: batch_train_labels}
                                        # graph_dict['batch_size']: batch_size}
                        else:
                            print ('Using the new RNN State')
                            feed_dict= {graph_dict['x']: batch_train_dataset, 
                                        graph_dict['y']: batch_train_labels,
                                        graph_dict['init_state'] : new_hid_layer_state}

                        bs, nwst, loss, opt, tp = sess.run([graph_dict['batch_size'],
                                                        graph_dict['new_state'],
                                                        graph_dict['loss_CE'],
                                                        graph_dict['optimizer'],
                                                        graph_dict['training_prediction']], 
                                                        feed_dict=feed_dict)
                        new_hid_layer_state = nwst

                        acc = self.accuracy(tp, batch_train_labels)

                        # print ('Batch size \n', bs)
                        # print ('')
                        # print ('New State \n', nwst)
                        # print ('')
                        # print ('Avg Loss \n',  loss)
                        # print ('')
                        # print ('optimizer \n', opt)
                        # print ('')
                        # print ('training_prediction \n', tp)
                        # print ('')
                        print ('accuracy \n', acc)
                        print ('')
                        print ('')
                        print ('popopopopopopopoop')
                        print ('')
                        print ('')



        
obj_Train = Train()
# print (obj_Train.vocab_size)
# print (obj_Model.vocab_size)
graph_dict =  dynamic_RNN_model(num_hid_units = 3,
                                vocab_size = obj_Train.vocab_size,
                                momentum = 0.9,
                                learning_rate = 0.01)
# print (graph_dict['batch_size'])
obj_Train.train_network(graph_dict)



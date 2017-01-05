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
        dictionary_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/dictionary.txt'
        self.vocab_size = len(corpora.Dictionary.load_from_text(dictionary_dir))
        self.train_batch_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/training_batch/'
        self.valid_batch_dir = '/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Word-Search-NNets/Word-Nets/crossvalid_batch/'

    def accuracy(self, predictions, labels, labels_one_hot = None):
        # The input labels are a One-Hot Vector
        if labels_one_hot:
            return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
                  / predictions.shape[0])
        else:
            return (100.0 * np.sum(np.argmax(predictions, 1) == np.reshape(labels, [-1]))
                  / predictions.shape[0])
        

    def train_network(self, graph_dict, num_batches):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            
            aa = np.array([[1,2,3,4,5], [1,3,4,0,5]])  # , [2,3,4,5,6], [2,1,5,4,3]
            bb = np.array([[2,3,4,5,2], [3,4,0,5,2]])  # , [3,4,5,6,2], [1,5,4,3,2]
            cc = np.array([[1,4,2,3,4,0], [1,3,4,3,5,0]])
            dd = np.array([[4,2,3,4,0,0], [3,4,3,5,0,0]])

            ee = np.array([[1,3,2,4,5], [2,3,4,0,5]]) 
            ff = np.array([[3,2,4,5,0], [3,4,0,5,0]])
            epochs = 1
            cdoc = np.random.choice(np.arange(20), 5) # Randomly select group of 5 batches from the cross valid dataset to test after every 20 batches
            for epoch in np.arange(epochs):
                new_hid_layer_state = None
#                 print (graph_dict)

                for m,n in [[aa,bb],[cc,dd]]:#np.arange(2):  # no in np.arange(num_batches):
                    # with open(self.train_batch_dir+'batch'+str(no)+'.pickle', 'rb') as f:
                    #     dataset = pickle.load(f)
                        
                        # batch_train_dataset = dataset['batch_train_dataset']
                        # batch_train_labels = dataset['batch_train_labels']

                        print (m)
                        print ('')
                        print (n)
                        print ('')
                        batch_train_dataset = m
                        batch_train_labels = n


                        if not new_hid_layer_state: 
                            print ('Using the Default RNN State')
                            feed_dict= {graph_dict['x']: batch_train_dataset, 
                                        graph_dict['y']: batch_train_labels}
                                        # graph_dict['batch_size']: batch_size}
                        else:
                            print ('Using the New RNN State')
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

                        print ('accuracy \n', acc)
                        print ('')
                        print ('popopopopopopopoop')
                        print ('')



                    # if (num_batches%20 ==0 and num_batches!=0):
                    #     print ('Evaluating cross validation dataset ')
                    #     for cdoc_no in cdoc:
                    #         with open(self.valid_batch_dir+'batch'+str(cdoc_no)+'.pickle', 'rb') as f1:
                    #             dataset = pickle.load(f)
                            
                    #             batch_valid_dataset = dataset['batch_valid_dataset']
                    #             batch_valid_labels = dataset['batch_valid_labels']

                    #             # We use the same weights that have been involved with the change on the cross validation dataset
                    #             feed_dict= {graph_dict['x']: batch_valid_dataset, 
                    #                         graph_dict['y']: batch_valid_labels,
                    #                         graph_dict['init_state'] : new_hid_layer_state}

                    #             bs, nwst, loss, opt, tp = sess.run([graph_dict['batch_size'],
                    #                                     graph_dict['new_state'],
                    #                                     graph_dict['loss_CE'],
                    #                                     graph_dict['optimizer'],
                    #                                     graph_dict['training_prediction']], 
                    #                                     feed_dict=feed_dict)

                    #             print ('accuracy \n', acc)
                    #             print ('')
                    #             print ('popopopopopopopoop')
                    #             print ('')

        
obj_Train = Train()
# print (obj_Train.vocab_size)
# print (obj_Model.vocab_size)
graph_dict =  dynamic_RNN_model(num_hid_units = 3,
                                vocab_size = obj_Train.vocab_size,
                                momentum = 0.9,
                                learning_rate = 0.01)
# print (graph_dict['batch_size'])
obj_Train.train_network(graph_dict, num_batches=2)



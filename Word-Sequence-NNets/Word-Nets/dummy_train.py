import numpy as np
import matplotlib.pyplot as plt
import time
import os
import urllib.request
from six.moves import cPickle as pickle
import tensorflow as tf
from tensorflow.models.rnn.ptb import reader
from gensim import corpora


from dummy_model import  dynamic_RNN_model


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
        

    def train_network(self, graph_dict):
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            
            aa = np.array([[1,2,3,4,5], [1,3,4,0,0], [1,3,4,0,0]])  # , [2,3,4,5,6], [2,1,5,4,3]
            bb = np.array([[2,3,4,5,6], [3,4,6,0,0], [3,4,6,0,0]])  # , [3,4,5,6,2], [1,5,4,3,2]
            ab = np.array([5,3,3])
            cc = np.array([[1,4,2,3,4,5], [1,3,4,0,0,0]])
            dd = np.array([[4,2,3,4,5,6], [3,4,6,0,0,0]])
            cd = np.array([6,3])

            # aa = np.array([[1,2,3,4,5], [1,3,4,2,4]])  # , [2,3,4,5,6], [2,1,5,4,3]
            # bb = np.array([[2,3,4,5,6], [3,4,2,4,6]])  # , [3,4,5,6,2], [1,5,4,3,2]
            # ab = np.array([5,5])
            # cc = np.array([[1,4,2,3,4,5], [1,3,4,2,3,4]])
            # dd = np.array([[4,2,3,4,5,6], [3,4,2,3,4,6]])
            # cd = np.array([6,6])

            ee = np.array([[1,3,2,3,4,5], [2,3,4,5,0,0]]) 
            ff = np.array([[3,2,3,4,5,6], [3,4,5,6,0,0]])
            ef = np.array([6,4])
            epochs = 4
            cdoc = np.random.choice(np.arange(20), 5) # Randomly select group of 5 batches from the cross valid dataset to test after every 20 batches
            for epoch in np.arange(epochs):
                training_loss = None
#                 print (graph_dict)
                new_state_ = None
                for no, (m,n,o) in enumerate([[aa,bb,ab],[cc,dd,cd]]):#np.arange(2):  # no in np.arange(num_batches):
                    batch_train_dataset = m
                    batch_train_labels = n
                    batch_train_lenarr = o

                    feed_dict= {graph_dict['x']: batch_train_dataset, 
                                graph_dict['y']: batch_train_labels,
                                graph_dict['x_lenarr']: batch_train_lenarr}
        
                    if new_state_ is not None:
                        print ('Using the new RNN State')
                        feed_dict[graph_dict['init_state']] = new_state_

                    x_, y_, x_lenarr_, batch_size_, init_state_, new_state_, rnn_output_, hid_to_ouptut_layer_, output_state_, softmax_opt_, loss_CE_, y_reshaped_, mask_, masked_loss_, mean_loss_by_example_, mean_loss_, optimizer_, training_prediction_ = \
                    sess.run([graph_dict['x'], 
                                graph_dict['y'], 
                                graph_dict['x_lenarr'], 
                                graph_dict['batch_size'],
                                graph_dict['init_state'],
                                graph_dict['new_state'],
                                graph_dict['rnn_outputs'],
                                graph_dict['hid_to_ouptut_layer'],
                                graph_dict['output_state'],
                                graph_dict['softmax_opt'],
                                graph_dict['loss_CE'],
                                graph_dict['y_reshaped'],
                                graph_dict['mask'],
                                graph_dict['masked_loss'],
                                graph_dict['mean_loss_by_example'], 
                                graph_dict['mean_loss'],
                                graph_dict['optimizer'],
                                graph_dict['prediction']], 
                                feed_dict=feed_dict)

                    # training_loss += loss


                    # acc = self.accuracy(tp, batch_train_labels)
                    print_output(x_, y_, x_lenarr_, batch_size_, init_state_, new_state_, rnn_output_, hid_to_ouptut_layer_, output_state_, softmax_opt_, loss_CE_, y_reshaped_, mask_, masked_loss_, mean_loss_by_example_, mean_loss_, training_prediction_)

                    if (no==1):
                        print ('crossvalid crossvalid crossvalid, crossvalid crossvalid crossvalid, crossvalid crossvalid crossvalid')
                        new_valid_state_ = None
                        for no_c, (m_c,n_c,o_c) in enumerate([[ee,ff,ef]]):
                            
                            batch_valid_dataset = m_c
                            batch_valid_labels = n_c
                            batch_valid_lenarr = o_c
                            
                            if new_valid_state_ is not None:
                                feed_dict={graph_dict['x']: batch_valid_dataset,
                                           graph_dict['x_lenarr']: batch_valid_lenarr,
                                           graph_dict['init_state']: new_valid_state_}
                            else:
                                feed_dict={graph_dict['x']: batch_valid_dataset,
                                        graph_dict['x_lenarr']: batch_valid_lenarr}

                            valid_prediction_, new_valid_state_ = sess.run([graph_dict['prediction'],
                                                                            graph_dict['new_state']], 
                                                                            feed_dict)
                            print (valid_prediction_)

def print_output(x_, y_, x_lenarr_, batch_size_, init_state_, new_state_, rnn_output_, hid_to_ouptut_layer_, output_state_, softmax_opt_, loss_CE_, y_reshaped_, mask_, masked_loss_, mean_loss_by_example_, mean_loss_, training_prediction_):
    print ('x = ', x_.shape , '\n',  x_)
    print ('')
    print ('y = ', x_.shape, '\n', y_)
    print ('')
    print ('x_lenarr = ', x_lenarr_)
    print ('')
    print ('batch_size = \n', batch_size_)
    print ('')
    print ('init_state = \n', init_state_)
    print ('')
    print ('new_state = \n',  new_state_)
    print ('')
    print ('rnn_output = ', rnn_output_.shape, '\n', rnn_output_)
    print ('')
    print ('hid_to_ouptut_layer = ', hid_to_ouptut_layer_.shape, '\n', hid_to_ouptut_layer_)
    print ('')
    print ('output_state = ', output_state_.shape, '\n', output_state_)
    print ('')
    print ('softmax_opt = ', softmax_opt_.shape, '\n', softmax_opt_)
    print ('')
    print ('loss_CE = ', loss_CE_.shape, '\n', loss_CE_)
    print ('')
    print ('y_reshaped = ', y_reshaped_.shape, '\n',  y_reshaped_)
    print ('')
    print ('mask = ', mask_.shape, '\n',  mask_)
    print ('')
    print ('masked_loss = ', masked_loss_.shape, '\n',  masked_loss_)
    print ('')
    print ('mean_loss_by_example = ', mean_loss_by_example_.shape, '\n', mean_loss_by_example_)
    print ('')
    print ('mean_loss = ', mean_loss_)
    print ('')
    print ('training_prediction = \n', training_prediction_)
    print ('')
    print ('')
    print ('popopopopopopopooppopopopopopopopooppopopopopopopopooppopopopopopopopoop')
    print ('')
    print ('')

obj_Train = Train()
graph_dict =  dynamic_RNN_model(vocab_size = 7)
obj_Train.train_network(graph_dict)

        
# obj_Train = Train()
# # print (obj_Train.vocab_size)
# # print (obj_Model.vocab_size)
# graph_dict =  dynamic_RNN_model(num_hid_units = 3,
#                                 vocab_size = obj_Train.vocab_size,
#                                 momentum = 0.9,
#                                 learning_rate = 0.01)
# # print (graph_dict['batch_size'])
# obj_Train.train_network(graph_dict, num_batches=2)



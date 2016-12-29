import numpy as np
import matplotlib.pyplot as plt
import time
import os
import urllib.request

import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

from model import dynamic_RNN_model




def train_network(graph_dict):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        
#         x_new = np.array([[1,4,2,2],[1,4,3,0]])
#         y_new = np.array([[4,2,2,5],[4,3,0,5]])
        
        training_data = np.array([[1,2,3,4], [1,3,4,0], [1,4,2,2], [1,4,3,0]])
        training_labels = np.array([[2,3,4,5], [3,4,0,5], [4,2,2,5], [4,3,0,5]])
        epochs = 10
        for epoch in np.arange(epochs):
            new_hid_layer_state = None
            for i in [2,4]:
    #         print (training_data[0:2,:])
                batch_data = training_data[i-2:i,:]
                batch_labels = training_labels[i-2:i,:]

                batch_size=len(batch_data)
                num_hidden_layer = 3
                num_classes = 6, 
                num_sequences = 4

    #             print (batch_data)
    #             print (batch_labels)

    #             print (batch_size)
                if not new_hid_layer_state: 
                    feed_dict= {graph_dict['x']: batch_data, 
                                graph_dict['y']: batch_labels}
                else:
                    print ('Using the new RNN State')
                    feed_dict= {graph_dict['x']: batch_data, 
                                graph_dict['y']: batch_labels, 
                                graph_dict['init_state'] : new_hid_layer_state}

                a, b, c, d, e, f, g, h, i, j, k = sess.run([graph_dict['embed_to_hid_wghts'],
                                         graph_dict['embed_to_hid_layer'],
                                         graph_dict['init_state'],
                                         graph_dict['rnn_outputs'],
                                         graph_dict['new_state'],
                                         graph_dict['hid_to_output_wght'],
                                         graph_dict['output_bias'],
                                         graph_dict['hid_to_ouptut_layer'],
                                         graph_dict['output_state'],
                                         graph_dict['loss_CE'],
                                         graph_dict['optimizer']], feed_dict=feed_dict)
                new_hid_layer_state = e

                print ('embed_to_hid_wghts \n', a)
                print ('')
                print ('embed_to_hid_layer \n', b)
                print ('')
                print ('init_state \n', c)
                print ('')
                print ('rnn_outputs \n', d)
                print ('')
                print ('new_state \n', e)
                print ('')
                print ('hid_to_output_wght \n', f)
                print ('')
                print ('output_bias \n', g)
                print ('')
                print ('hid_to_ouptut_layer \n', h)
                print ('')
                print ('output_state \n', i)
                print ('')
                print ('loss_CE \n', j)
                print ('')
                print ('optimizer \n', k)
                print ('')
                print ('')
                print ('popopopopopopopoop')
                print ('')
                print ('')

        
graph_dict = dynamic_RNN_model()
train_network(graph_dict)


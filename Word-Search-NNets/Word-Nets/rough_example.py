import numpy as np
import matplotlib.pyplot as plt
import time
import os
import urllib.request

import tensorflow as tf
from tensorflow.models.rnn.ptb import reader
# %matplotlib inline



def reset_graph():  # Reset the graph
    print ('Resetting The Graph')
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

# vacab_size = 6
def method_1():
	# batch_size = 1
    # batch_size = 2
    num_hid_units = 3
    vocab_size = 6

    reset_graph()
    x = tf.placeholder(tf.int32, shape = [None, None], name='input_placeholder')
    batch_size = tf.shape(x)[0]

    embed_to_hid_wghts = tf.get_variable('embedding_matrix', [vocab_size, num_hid_units])
    embed_to_hid_layer = tf.nn.embedding_lookup(embed_to_hid_wghts, x)

    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_hid_units, state_is_tuple=True)
    init_state = rnn_cell.zero_state(batch_size, tf.float32)
    rnn_outputs, new_state = tf.nn.dynamic_rnn(
                                        cell=rnn_cell,
                                        # sequence_length=X_lengths,
                                        initial_state=init_state,
                                        inputs=embed_to_hid_layer)

    return dict(
        x=x,
        batch_size = batch_size,
        embed_to_hid_wghts = embed_to_hid_wghts,
        embed_to_hid_layer = embed_to_hid_layer,
        init_state = init_state,
        # new_state = new_state
    )



def method_2(graph_dict):
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())

        aa = np.array([[1,2,3,4,5,1], [1,3,4,0,5,1], [5,4,3,2,1,1]])
        bb = np.array([[1,2,3,4,5], [1,3,4,0,5]])
        for i in [aa,bb]:
            feed_dict= {graph_dict['x']: i}
            b, ethw, ethl, ist = sess.run([graph_dict['batch_size'],  # ethw,ethl,nst
                                            graph_dict['embed_to_hid_wghts'],
                                            graph_dict['embed_to_hid_layer'],  
                                            graph_dict['init_state'],
                                            # graph_dict['new_state']
                                            ], feed_dict=feed_dict)
            print ('batch_s \n', b)
            print ('')
            print ('embed_to_hid_wghts \n', ethw)
            print ('')
            print ('embed_to_hid_layer \n', ethl)
            print ('')
            print ('init_state \n', ist)
            print ('')
            # print ('new_state \n', nst)
            # print ('')
            print ('')

graph_dict = method_1()
method_2(graph_dict)
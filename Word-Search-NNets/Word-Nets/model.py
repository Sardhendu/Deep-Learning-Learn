import numpy as np
import matplotlib.pyplot as plt
import time
import os
import urllib.request

import tensorflow as tf
from tensorflow.models.rnn.ptb import reader

# %matplotlib inline



def reset_graph():  # Reset the graph
    if 'sess' in globals() and sess:
        sess.close()
    tf.reset_default_graph()

# vacab_size = 6
def dynamic_RNN_model(
    num_hid_units = 3,
    vocab_size = 7,
    momentum = 0.9,
    learning_rate = 0.001
    ):
    print ('The num of hidden unit is: ', num_hid_units)
    print ('The Vocab size is: ', vocab_size)
    print ('The momentum is: ', momentum)
    print ('The learning_rate is: ', learning_rate)
    
    
    num_classes = vocab_size

    reset_graph()
    
    x = tf.placeholder(tf.int32, shape = [None, None], name='input_placeholder')
    y = tf.placeholder(tf.int32, shape = [None, None], name='output_placeholdr')
    x_lenarr = tf.placeholder(tf.float32, shape = [None], name='output_placeholdr')
    batch_size = tf.shape(x)[0]
    

    # ENBEDDING(INPUT) LAYER OPERATION
    # Creating an Embedding matrix with a random weight for all vacab to hidden_matrix
    embed_to_hid_wghts = tf.get_variable('embedding_matrix', [vocab_size, num_hid_units])
    embed_to_hid_layer = tf.nn.embedding_lookup(embed_to_hid_wghts, x)
    print ('The shape of embed_to_hid_wghts is: ', embed_to_hid_wghts.get_shape())
    print ('The shape of embed_to_hid_layer is: ', embed_to_hid_layer.get_shape())


    # HIDDEN LAYER OPERATION
    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_hid_units, state_is_tuple=True)
    init_state = rnn_cell.zero_state(batch_size, tf.float32)  # Each sequence will hava a state that it passes to its next sequence
    rnn_outputs, new_state = tf.nn.dynamic_rnn(cell=rnn_cell,
                                               sequence_length=x_lenarr,
                                               initial_state=init_state,
                                               inputs=embed_to_hid_layer,
                                               dtype=tf.float32)
    

    # OUTPUT LAYER OPERATION
    # Initialize the weight and biases for the output layer. We use variable scope because we would like to share the weights 
    with tf.variable_scope('output_layer'):
        hid_to_output_wght = tf.get_variable('hid_to_output_wght', [num_hid_units, num_classes],
                                            initializer = tf.random_normal_initializer())
        output_bias = tf.get_variable('output_bias', [num_classes],
                                      initializer=tf.random_normal_initializer())
    
    rnn_outputs = tf.reshape(rnn_outputs, [-1, num_hid_units])
    hid_to_ouptut_layer = tf.matmul(rnn_outputs, hid_to_output_wght) +  output_bias  
    # Also use tf.batch_matmul(rnn_outputs, hid_to_output_wght) +  output_bias  
    output_state = tf.nn.softmax(hid_to_ouptut_layer, name=None)
 
    
    # CALCULATING LOSS, OPTIMIZING THE COST FUNCTION, MEASURING ACCURACY
    y_reshaped = tf.reshape(y, [-1])
    softmax_opt = tf.nn.sparse_softmax_cross_entropy_with_logits(hid_to_ouptut_layer, y_reshaped)
    loss_CE = tf.reduce_mean(softmax_opt)
    
    # # MASK THE LOSES
    mask = tf.sign(tf.to_float(y_reshaped))  
    masked_loss = mask * loss_CE
 
    # # Bring back to [B, T] shape
    masked_loss = tf.reshape(masked_loss,  tf.shape(y))

    # # Calculate mean loss
    mean_loss_by_example = tf.reduce_sum(masked_loss, reduction_indices=1) / x_lenarr
    mean_loss = tf.reduce_mean(mean_loss_by_example)

    # The sparse_softmax uses dtype as int32 or int64

    # OPTIMIZE
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_CE)
#     optimizer = tf.train.MomentumOptimizer(learning_rate, 
#                                             momentum, 
#                                             use_locking=False, 
#                                             name='Momentum', 
#                                             use_nesterov=True).minimize(loss_CE)

    # Returns graph objects
    return dict(
        x=x,
        y=y,
        x_lenarr=x_lenarr,
        batch_size = batch_size,
        init_state = init_state,
        new_state = new_state,
        rnn_outputs = rnn_outputs,
        hid_to_ouptut_layer = hid_to_ouptut_layer,
        output_state = output_state,
        softmax_opt = softmax_opt,
        loss_CE = loss_CE,
        y_reshaped = y_reshaped,
        mask = mask,
        masked_loss = masked_loss,
        mean_loss_by_example = mean_loss_by_example,
        mean_loss = mean_loss,
        optimizer = optimizer,
        training_prediction = output_state
    )

# Model()
    
"""
Notes:
    1. embed_to_hid_wghts = tf.get_variable('embedding_matrix', [vocab_size, num_hid_units]):
       embed_to_hid_layer = tf.nn.embedding_lookup(embed_to_hid_wghts, x)
        # Normally we convert the input vector into a one hot matrix and then multiply it to the embedded weights, When we do so, we get the same embed weight corresponding to 1's in the one-hot vector but in a different shape. The above operation does all that in a single shot. Basically, embed_to_hid_wghts defines a matrix with weights going form all vacab to hiddenunits,and embed_to_hid_layer pulls the vectors from embedding_matrix(embed_to_hid_wghts) corresponding to the idices entries of x for all the batch. So the matrix embed_to_hid_layer = [batch_size x num_sequences x num_hid_units]

    2. The variable rnn_output is a Tensor of shape of [Batch_size x num_sequence x num_hid_units] and,
       The hid_to_output_wght is in the shape of [num_hid_units x num_classes]
       And We want an output with shape [Batch_size x num_sequence x num_classes]
       We horizontlly stack all the batches to form a matrix of [(Batch_size x num_sequence]) x num_classes]

    3. sparse_softmax_cross_entropy_with_logits automatically converts the y's into on hot vectors and perform 
       the softmax operation When using softmax_cross_entropy_with_logits, we have to first convert the y's into one-hot vector
"""
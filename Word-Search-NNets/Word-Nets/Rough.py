def dynamic_RNN_model(
    batch_size = 2,
    num_hid_units = 3,
    num_classes = 6,
    num_sequences = 4,
    momentum = 0.9,
    learning_rate = 0.5):

    vocab_size = num_classes
    
    reset_graph()
    
    x = tf.placeholder(tf.int32, shape = [batch_size, num_sequences], name='input_placeholder')
    y = tf.placeholder(tf.int32, shape = [batch_size, num_sequences], name='output_placeholder')

    # ENBEDDING(Input) LAYER TO HIDDEN LAYER OPERATION
    # Creating an Embedding matrix with a random weight for all vacab to hidden_matrix
    embed_to_hid_wghts = tf.get_variable('embedding_matrix', [vocab_size, num_hid_units])
    # Normally we convert the input index into a one hot matrix and then multiply it to the embedded weights, When we do so, we get the same embed weight corresponding to 1's in the one-hot vector but in a different shape. The below operation does all that in a single shot.
    embed_to_hid_layer = tf.nn.embedding_lookup(embed_to_hid_wghts, x)

    # HIDDEN LAYER OPERATION
    rnn_cell = tf.nn.rnn_cell.LSTMCell(num_hid_units, state_is_tuple=True)
    init_state = rnn_cell.zero_state(batch_size, tf.float32)  # Each sequence will hava a state that it passes to its next sequence
    rnn_outputs, new_state = tf.nn.dynamic_rnn(
                                        cell=rnn_cell,
                                        # sequence_length=X_lengths,
                                        initial_state=init_state,
                                        inputs=embed_to_hid_layer)
    
    # Initialize the weight and biases for the output layer. We use variable scope because we would like to share the weights 
    with tf.variable_scope('output_layer'):
        hid_to_output_wght = tf.get_variable('hid_to_output_wght', 
                                                 [num_hid_units, num_classes], 
                                                 initializer = tf.random_normal_initializer())
        output_bias = tf.get_variable('output_bias',
                                      [num_classes],
                                      initializer = tf.random_normal_initializer())
    

    # OUTPUT LAYER OPERATION
    # The variable rnn_output is a Tensor of shape of [Batch_size x num_sequence x num_hid_units] and,
    # The hid_to_output_wght is in the shape of [num_hid_units x num_classes]
    # And We want an output with shape [Batch_size x num_sequence x num_classes]
    # We horizontlly stack all the batches to form a matrix of [(Batch_size x num_sequence]) x num_classes]
    rnn_outputs = tf.reshape(rnn_outputs, [-1, num_hid_units])  
    hid_to_ouptut_layer = tf.matmul(rnn_outputs, hid_to_output_wght) +  output_bias
    output_state = tf.nn.softmax(hid_to_ouptut_layer, name=None)
 
    
    # CALCULATING LOSS and OPTIMIZING THE COST FUNCTION
    loss_CE = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(hid_to_ouptut_layer, tf.reshape(y, [-1])))
    # sparse_softmax_cross_entropy_with_logits automatically converts the y's into on hot vectors and perform the softmax operation
    # When using softmax_cross_entropy_with_logits, we have to first convert the y's into one-hot vector
    # The sparse_softmax uses dtype as int32 or int64
    optimizer = tf.train.MomentumOptimizer(learning_rate, 
                                            momentum, 
                                            use_locking=False, 
                                            name='Momentum', 
                                            use_nesterov=True).minimize(loss_CE)
        
    
    # Returns a graph object
    return dict(
        x=x,
        y=y,
        embed_to_hid_wghts = embed_to_hid_wghts,
        hid_to_output_wght = hid_to_output_wght,
        init_state = init_state,
        new_state = new_state,
        loss_CE = loss_CE,
        optimizer = optimizer,
        training_prediction = output_state
    )









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

                a, b, c, e, j, k, l = sess.run([graph_dict['embed_to_hid_wghts'],
                                          graph_dict['hid_to_output_wght'],
                                          graph_dict['init_state'],
                                         graph_dict['new_state'],
                                         graph_dict['loss_CE'],
                                         graph_dict['optimizer'],
                                         graph_dict['training_prediction']], feed_dict=feed_dict)
                new_hid_layer_state = e

                print ('embed_to_hid_wghts \n', a)
                print ('')
                print ('hid_to_output_wght \n', b)
                print ('')
#                 print ('init_state \n', c)
#                 print ('')
# #                 print ('rnn_outputs \n', d)
# #                 print ('')
#                 print ('new_state \n', e)
#                 print ('')
# #                 print ('hid_to_output_wght \n', f)
# #                 print ('')
# #                 print ('output_bias \n', g)
# #                 print ('')
# #                 print ('hid_to_ouptut_layer \n', h)
# #                 print ('')
# #                 print ('output_state \n', i)
# #                 print ('')
#                 print ('loss_CE \n', j)
#                 print ('')
#                 print ('optimizer \n', k)
#                 print ('')
#                 print ('training_prediction \n', l)
#                 print ('')
#                 print ('')
                print ('popopopopopopopoop')
                print ('')
                print ('')

        
graph_dict = dynamic_RNN_model()
train_network(graph_dict)


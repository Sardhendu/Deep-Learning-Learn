Notes: Understanding the Network:


File: frwd_propogate, Method: fprop, Formula: 
Let assume the below inputs for easy representation.

1. total words in vocabulary (vocab_size)= 12
2. number of neurons in embedding layer (no_embed_unit) = 3
3. no of training examples (batch_size) = 3
4. number of neurons in hidden layer 2 (no_hidden_unit) = 20
5. baise for the hidden layer 2 (hid_bias)= (20 x 1)

training_data - per training sample has three words w1,w2,w3, each cell value id the word vocab index:
input_batch =
          W1  W2  W3
     T1  [0,  1,  2],
     T2  [3,  4,  5],
     T3  [6,  7,  8]

Weights of each word in the vocabulary to each neuron
# word_to_embed_wghts (11 x 3)=
         N1           N2        N3
	v0  [0.0162,    0.7783,    0.1202],
    v1  [0.4822,    0.3084,    0.2853],
    v2  [0.5349,    0.5013,    0.6023],
    v3  [0.5364,    0.7758,    0.7473],
    v4  [0.1828,    0.2145,    0.7754],
    v5  [0.3712,    0.6758,    0.8971],
    v6  [0.1369,    0.3729,    0.8267],
    v7  [0.8031,    0.3043,    0.8206],
    v8  [0.0841,    0.8010,    0.6298],
    v9  [0.4896,    0.0067,    0.9258],
    v10 [0.3357,    0.0360,    0.0614],
    v11 [0.8424,    0.0281,    0.8296]

	# Flatten the training data:
	np.reshape(input_batch,(1,no_words_ngram*batch_size))[0]
			 =  array([0, 1, 2, 3, 4, 5, 6, 7, 8])

	# Get only the rows pertainig to indices in the input_batch
	word_to_embed_wghts[np.reshape(input_batch,(1,no_words_ngram*batch_size))[0]] = 
			               N1        N2       N3
		(W1 T1) --> v0 [ 0.0162,  0.7783,  0.1202],
		(W1 T2) --> v1 [ 0.4822,  0.3084,  0.2853],
		(W1 T3) --> v2 [ 0.5349,  0.5013,  0.6023],
		(W2 T1) --> v3 [ 0.5364,  0.7758,  0.7473],
		(W2 T2) --> v4 [ 0.1828,  0.2145,  0.7754],
		(W2 T3) --> v5 [ 0.3712,  0.6758,  0.8971],
		(W3 T1) --> v6 [ 0.1369,  0.3729,  0.8267],
		(W3 T2) --> v7 [ 0.8031,  0.3043,  0.8206],
		(W3 T3) --> v8 [ 0.0841,  0.801 ,  0.6298]

	# Transpose the above matrix
	np.transpose(word_to_embed_wghts[np.reshape(input_batch,(1,no_words_ngram*batch_size))[0]])
	N1 [ 0.0162, 0.4822, 0.5349, 0.5364, 0.1828, 0.3712, 0.1369, 0.8031, 0.0841]
	N2 [ 0.7783, 0.3084, 0.5013, 0.7758, 0.2145, 0.6758, 0.3729, 0.3043, 0.801 ]
	N3 [ 0.1202, 0.2853, 0.6023, 0.7473, 0.7754, 0.8971, 0.8267, 0.8206, 0.6298]

	# The final state of the Embedding layer: Each Cell shows the weight of each word of a training set to a particular neuron.
	
# ebmed_layer_state (9 x 3) =
	np.reshape(np.transpose(word_to_embed_wghts[np.reshape(input_batch,(1,no_words_ngram*batch_size))[0]]), (no_words_ngram*no_embed_unit, batch_size), order = 'F')
    			T1         T2        T3
    W1 --> N1 [0.0162,  0.5364,  0.1369]
    W1 --> N2 [0.7783,  0.7758,  0.3729]
    W1 --> N3 [0.1202,  0.7473,  0.8267]
    W2 --> N4 [0.4822,  0.1828,  0.8031]
    W2 --> N5 [0.3084,  0.2145,  0.3043]
    W2 --> N6 [0.2853,  0.7754,  0.8206]
    W3 --> N7 [0.5349,  0.3712,  0.0841]
    W4 --> N8 [0.5013,  0.6758,  0.801 ]
    W3 --> N9 [0.6023,  0.8971,  0.6298]


# inputs_to_hid (20 x 3) = 
	--> inputs to hid2 should be the dot product of the state of the embedding layer (ebmed_layer_state) and the weights from each nueron to the hid2_layer added to the bias vector added to all the weight vectors similar to the formula y = b0 + b1x1 + b2x2+ ...... where b0, b1, b2 ... are the weights. The y represents the training row in this case th 

    np.dot(np.transpose(embed_to_hid_wgths), ebmed_layer_state) + np.tile(hid_bias, (1,batch_size))

    embed_to_hid_wgths -> (9 x 20)
    ebmed_layer_state -> (9 x 3)
    hid_bias -> (20 x 1)
    np.tile(hid_bias, (1,batch_size)) -> (20 x 3)   repeats the column 3 times

    inputs_to_hid (20 x 3) ->  [(20 x 9) x (9 x 3)] +  [(20 x 3)] 


    embed_to_hid_wgths.T = 
                N1       N2     N3     N4     N5     N6     N7     N8     N9   
        h2_N1   w11      w21    w31    w41    w51    w61    w71    w81    w91  
    	h2_N2   w12      w22    w32    w42    w52    w62    w72    w82    w92
    	h2_N3   w13      w23    w33    w43    w53    w63    w73    w83    w93
    	h2_N4   w14      w24    w34    w44    w54    w64    w74    w84    w94
    	.        .       .      .      .      .      .      .      .      .   
    	.        .       .      .      .      .      .      .      .      .    
    	h2_N19  w1,19   w2,19  w3,19  w4,19  w5,19  w6,19  w7,19  w8,19  w9,19
    	h2_N20  w1,20   w2,20  w3,20  w4,20  w5,20  w6,20  w7,20  w8,20  w9,20

    ebmed_layer_state =
	         T1     T2     T3
	    N1   x11    x12    x13
	    N2   x21    x22    x23
	    N3   x31    x32    x33
	    N4   x41    x42    x43 
	    N5   x51    x52    x53
	    N6   x61    x62    x63
	    N7   x71    x72    x73
	    N8   x81    x82    x83
	    N9   x91    x92    x93

	hid_bais =
	            T1       T2     T3   
        h2_b1   b11      b12    b13  
    	h2_b2   b21      b22    b23  
    	h2_b3   b31      b32    b33  
    	h2_b4   b41      b42    b43    
    	.       .       .      .    
    	.       .       .      .    
    	h2_b19  w1,19   w2,19  w3,19 
    	h2_b20  w1,20   w2,20  w3,20 

    inputs_to_hid = (embed_to_hid_wgths.T x ebmed_layer_state) + hid2_bais
			   	        T1                      T2                        T3   
    h2_inp   b11+(w11*x11+w22*x22+..)  b11+(w11*x11+w22*x22+..) b11+(w11*x11+w22*x22+..)  
    .                   .                        .                        .    
    .                   .                        .                        .     
    h2_inp              .                        .                        .  

# hid_layer_state (20 x 3): 
	--> The state of the input layer is just the sigmoid function of the sum of squared error 

# inputs_to_softmax :
    inputs_to_softmax = np.dot(np.transpose(hid_to_outpt_wghts), hid_layer_state) + np.tile(outpt_bias, (1,batch_size))
    The same Formula of sum of squared error.
    
	# Subtract maximum. 
	# Remember that adding or subtracting the same constant from each input to a
	# softmax unit does not affect the outputs. Here we are subtracting maximum to
	# make all inputs <= 0. This prevents overflows when computing their
	# exponents.

	inputs_to_softmax = inputs_to_softmax - np.tile(np.max(inputs_to_softmax, axis=0), (vocab_size, 1));

	if vocab_size = 5 and batch_size = 4 and given below are the weights- inputs to the softmax function
	inputs_to_softmax = 
            T1         T2         T3         T4
	v1   [0.9009,    0.0449,    0.5118,    0.5521],
    v2   [0.1771,    0.6911,    0.5117,    0.8446],
    v3   [0.7151,    0.1719,    0.8982,    0.0265],
    v4   [0.8903,    0.8399,    0.3048,    0.1368],
    v5   [0.3532,    0.7981,    0.3180,    0.6674]

    np.max(inputs_to_softmax, axis=0) -> finds the max from each column
         [ 0.9009,  0.8399,  0.8982,  0.8446]

    np.tile(np.max(inputs_to_softmax, axis=0), (vocab_size, 1))
         [ 0.9009,  0.8399,  0.8982,  0.8446],
         [ 0.9009,  0.8399,  0.8982,  0.8446],
         [ 0.9009,  0.8399,  0.8982,  0.8446],
         [ 0.9009,  0.8399,  0.8982,  0.8446],
         [ 0.9009,  0.8399,  0.8982,  0.8446]

	inputs_to_softmax = inputs_to_softmax - np.tile(np.max(inputs_to_softmax, axis=0), (vocab_size, 1))
	        T1        T2       T3       T4
    v1   [ 0.    , -0.795 , -0.3864, -0.2925],
    v2   [-0.7238, -0.1488, -0.3865,  0.    ],
    v3   [-0.1858, -0.668 ,  0.    , -0.8181],
    v4   [-0.0106,  0.    , -0.5934, -0.7078],
    v5   [-0.5477, -0.0418, -0.5802, -0.1772]

    output_layer_state = np.exp(inputs_to_softmax)
       [ 1.        ,  0.45158123,  0.67949867,  0.74639525],
       [ 0.48490611,  0.86174145,  0.67943073,  1.        ],
       [ 0.83043967,  0.51273302,  1.        ,  0.44126927],
       [ 0.98945598,  1.        ,  0.55244577,  0.49272701],
       [ 0.57827832,  0.95906157,  0.5597864 ,  0.83761225]

# output_layer_state =  
    output_layer_state / np.tile(np.sum(output_layer_state, axis=0), vocab_size, 1)
            T1            T2           T3            T4
    v1 [ 0.25752752,  0.11930442,  0.19575541,  0.21216442],
    v2 [ 0.12487667,  0.22766572,  0.19573584,  0.28425211],
    v3 [ 0.21386107,  0.13546027,  0.288088  ,  0.12543172],
    v4 [ 0.25481215,  0.26419261,  0.159153  ,  0.14005869],
    v5 [ 0.14892258,  0.25337698,  0.16126774,  0.23809305]
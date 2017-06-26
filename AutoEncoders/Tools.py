from __future__ import division
from __future__ import print_function
import logging

import numpy as np
from NetworkDesign import  NetworkDesign

new_logging = True

if new_logging:
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="w",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
else:
    logging.basicConfig(level=logging.DEBUG, filename="logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    
    
def sigmoid(xIn):
    return 1/(1+np.exp(-xIn))

def klDivergence(rho, rho_hat):
    """
    :param rho: The sparsity Parameter, A very small value close to 0 probably 0.05
    :param rho_hat: The activation of a neuron
    :return: The divergence of rho_hat from rho
    """
    return ((rho * np.log(rho/rho_hat)) +
            (rho_hat * np.log((1-rho)/(1-rho_hat)))
        )



class SparseAutoEncoders():
    def __init__(self, numInpUnits, numHidUnits, numOutUnits, rho, lambda_, beta):
        """

        :param rho: The sparse parameter value
        :param lamda: The weight decay parameter value
        :param beta:
        
        :param numInpUnits: Number of hidden units
        :param numHidUnits: Number of Hidden activation units
        :param numOutUnits: Number of output units,
                            For an autoencoder this has to be equal
                            to the nunImpUnits
        :param rho:         The sparse parameter value
        :param lambda_:     The weight decay parameter value
        :param beta:        The penalty for high divergence.
        """
        
        self.rho = rho
        self.lambda_ = lambda_
        self.beta = beta

        self.numInpUnits = numInpUnits
        self.numHidUnits = numHidUnits
        self.numOutUnits = numOutUnits

    def forwardProp(self, input, activationType='sigmoid'):
        numInpUnits = self.numInpUnits
        numHidUnits = self.numHidUnits
        numOutUnits = self.numOutUnits
        
        
        logging.info('The shape of the input is: %s', input.shape)
        print ('The input is: ', input)
        a = NetworkDesign(numInpUnits=numInpUnits,
                          numHidUnits=numHidUnits,
                          numOutUnits=numOutUnits,
                          )

        a.initBias()
        a.initWeights()
        W1 = a.params['w'][0]
        W2 = a.params['w'][1]
        b1 = a.params['b'][0]
        b2 = a.params['b'][1]

        
        logging.info('The shape of the input to hidden weights is: %s', W1.shape)
        logging.info('The shape of the input to hidden bias is: %s', b1.shape)
        logging.info('The shape of the hidden to output weights is: %s', W2.shape)
        logging.info('The shape of the hidden to output bias is: %s', b2.shape)
        
        hidLayerState = sigmoid(np.dot(input, W1) + b1)
        logging.info('The shape of the hidden layer state is: %s', hidLayerState.shape)

        outLayerState = sigmoid(
                np.dot(hidLayerState, W2) + b2
        )
        logging.info('The shape of the output layer state is: %s', outLayerState.shape)
        
        # Compute the average activation rho_hat
        # Each Neuron's output is dot product of weights and input. So here we average the activation (hidden state)
        # of each neuron.
        rho_hat = np.sum(hidLayerState, axis=0) / input.shape[1]
        logging.info('The Average activation (rho_hat) shape (equals the number of Neurons) is: %s %s', rho_hat.shape,
                     rho_hat)
        rho = np.tile(self.rho, numHidUnits)
        divergence = klDivergence(rho, rho_hat)
        print (divergence)
        
        # Compute the cost function (minimum squared error)
        # In Auto-Encoders the input is the output, we just use the hidden layer to learn interesting representation
        # or weights for latent factors
        cost = (1/2*input.shape[1]) * pow(np.sum(outLayerState - input), 2) + \
               ((self.lambda_/2) * (np.sum(W1**2) + np.sum(W2**2))) + \
               (self.beta * np.sum(divergence))
        
        print (cost)
        
        
        # Backward Propagation The derivate term for rho_hat
        row_delta = np.tile(
                (- rho / rho_hat + (1 - rho) / (1 - rho_hat)),
                (input.shape[1], 1)
        )
        
        
b = SparseAutoEncoders(numInpUnits = 3,
                       numHidUnits = 2,
                       numOutUnits = 3,
                       rho = 0.05, lambda_=0.5, beta =0.3).forwardProp(input=np.array([[1,2,3]], dtype=float))
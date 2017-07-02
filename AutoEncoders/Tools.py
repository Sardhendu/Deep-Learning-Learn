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
        m,n = input.shape
        
        logging.info('The shape of the input is: %s', input.shape)
        print ('The input is: ', input)
        a = NetworkDesign(numInpUnits=numInpUnits,
                          numHidUnits=numHidUnits,
                          numOutUnits=numOutUnits,
                        )

        a.initBias()
        a.initWeights()
        inp_to_hid_wghts = a.params['w'][0]
        hid_to_out_wghts = a.params['w'][1]
        inp_to_hid_bias = a.params['b'][0]
        hid_to_out_bias = a.params['b'][1]

        
        logging.info('The shape of the input to hidden weights is: %s', inp_to_hid_wghts.shape)
        logging.info('The shape of the input to hidden bias is: %s', inp_to_hid_bias.shape)
        logging.info('The shape of the hidden to output weights is: %s', hid_to_out_wghts.shape)
        logging.info('The shape of the hidden to output bias is: %s', hid_to_out_bias.shape)
        
        hidLayerState = sigmoid(
                np.dot(input, inp_to_hid_wghts) + inp_to_hid_bias
        )
        logging.info('The shape of the hidden layer state is: %s', hidLayerState.shape)

        # We dont use the sigmoid activation for the output unit,since we are training an Autoencoder.
        outLayerState = np.dot(hidLayerState, hid_to_out_wghts) + hid_to_out_bias
        logging.info('The shape of the output layer state is: %s', outLayerState.shape)
        
        # Compute the average activation rho_hat
        # Each Neuron's output is dot product of weights and input. So here we average the activation (hidden state)
        # of each neuron.
        rho_hat = np.sum(hidLayerState, axis=0) / m
        logging.info('The Average activation (rho_hat) shape (equals the number of Neurons) is: %s %s', rho_hat.shape,
                     rho_hat)
        rho = np.tile(self.rho, numHidUnits)
        divergence = klDivergence(rho, rho_hat)
        print ('divergence for each hidden units: \n', divergence)
        
        # Compute the regularization parameter
        reg = np.sum(inp_to_hid_wghts**2) + np.sum(hid_to_out_wghts**2)
        logging.info('The reg is: %s %s', reg.shape, reg)
        
        # Compute the cost function (minimum squared error)
        # In Auto-Encoders the input is the output, we just use the hidden layer to learn interesting representation
        # or weights for latent factors
        cost = (1/2*m) * pow(np.sum(outLayerState - input), 2) + \
               ((self.lambda_/2) * reg) + \
               (self.beta * np.sum(divergence))
        
        print ('cost: \n', cost)
        
        
        # Backward Propagation The derivate term for divergence, The below is just the derivative term repeated
        delta_rho = np.tile(
                (- rho / rho_hat + (1 - rho) / (1 - rho_hat)),   # This is the derivative (gradient) of rho
                (m, 1)
        )
        logging.info('The shape of delta_rhois: %s', delta_rho.shape)
        print ('rho_delta: \n', delta_rho)
        
        
        # Compute the gradients for the Output state and hidden state
        delta_outLayer = -(input - outLayerState)
        logging.info('The shape of delta_outLayer is: %s', delta_outLayer.shape)
        delta_hidLayer = (np.dot(delta_outLayer, np.transpose(hid_to_out_wghts)) + \
                          + self.beta * delta_rho
                          )* sigmoid(hidLayerState) * (1 - sigmoid(hidLayerState))
        logging.info('The shape of delta_hidLayer is: %s', delta_hidLayer.shape)
        
        
        # Compute the weight gradients:
        delta_hid_to_out_wghts = (1 / m) * np.dot(np.transpose(hidLayerState),
                                                  delta_outLayer) + (self.lambda_ * hid_to_out_wghts)
        logging.info('The shape of delta_hid_to_out_wghts is: %s', delta_hid_to_out_wghts.shape)
        
        delta_inp_to_hid_wghts = (1 / m) * np.dot(np.transpose(input),
                                                  delta_hidLayer) + (self.lambda_ * inp_to_hid_wghts)
        logging.info('The shape of delta_inp_to_hid_wghts is: %s', delta_inp_to_hid_wghts.shape)
        
        
        
debug = True

if debug:
    b = SparseAutoEncoders(numInpUnits = 3,
                       numHidUnits = 2,
                       numOutUnits = 3,
                       rho = 0.05, lambda_=0.5, beta =0.3).forwardProp(input=np.array([[1,2,3]], dtype=float))
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
        
        # self.rho_val = rho
        self.lambda_ = lambda_
        self.beta = beta

        self.numInpUnits = numInpUnits
        self.numHidUnits = numHidUnits
        self.numOutUnits = numOutUnits
        
        # Repeat teh
        self.rho = np.tile(rho, self.numHidUnits)
    
    def setData(self, input):
        self.input = input
        self.m, self.n = input.shape
        logging.info('The shape of the self.input is: %s', input.shape)
        print('The self.input is: ', input)
        
    def initializeParams(self):
        a = NetworkDesign(numInpUnits=self.numInpUnits,
                          numHidUnits=self.numHidUnits,
                          numOutUnits=self.numOutUnits,
                          )
    
        a.initBias()
        a.initWeights()
        
        self.inp_to_hid_wghts = a.params['w'][0]
        self.hid_to_out_wghts = a.params['w'][1]
        self.inp_to_hid_bias = a.params['b'][0]
        self.hid_to_out_bias = a.params['b'][1]
    
        logging.info('The shape of the self.input to hidden weights is: %s', self.inp_to_hid_wghts.shape)
        logging.info('The shape of the self.input to hidden bias is: %s', self.inp_to_hid_bias.shape)
        logging.info('The shape of the hidden to output weights is: %s', self.hid_to_out_wghts.shape)
        logging.info('The shape of the hidden to output bias is: %s', self.hid_to_out_bias.shape)
      
        
    def sparsity(self, hidLayerState, numInputs):
        # Compute the average activation rho_hat
        self.rho_hat = np.sum(hidLayerState, axis=0) / numInputs
        logging.info('The Average activation (rho_hat) shape (equals the number of Neurons) is: %s %s',
                     self.rho_hat.shape, self.rho_hat)
        divergence = klDivergence(self.rho, self.rho_hat)
        print('divergence for each hidden units: \n', divergence)
        return divergence
    
    
    def deltaSparsity(self, numInputs):
        delta_rho = np.tile(
                (- self.rho / self.rho_hat + (1 - self.rho) / (1 - self.rho_hat)),  # This is the derivative (
                # gradient) of
                #  rho
                (numInputs, 1)
        )
        logging.info('The shape of delta_rhois: %s', delta_rho.shape)
        print('rho_delta: \n', delta_rho)
        return delta_rho
    
        
    def forwardProp(self, activationType='sigmoid'):
        
        # Compute the hidden layer state
        self.hidLayerState = sigmoid(
                np.dot(self.input, self.inp_to_hid_wghts) + self.inp_to_hid_bias
        )
        logging.info('The shape of the hidden layer state is: %s', self.hidLayerState.shape)

        # We don't use the sigmoid activation for the output unit,since we are training an Autoencoder.
        self.outLayerState = np.dot(self.hidLayerState, self.hid_to_out_wghts) + self.hid_to_out_bias
        logging.info('The shape of the output layer state is: %s', self.outLayerState.shape)

        # Compute the divergence
        divergence = self.sparsity(self.hidLayerState, numInputs=self.m)
        
        # Compute the regularization parameter
        reg = np.sum(self.inp_to_hid_wghts**2) + np.sum(self.hid_to_out_wghts**2)
        logging.info('The reg is: %s %s', reg.shape, reg)
        
        # Compute the cost function (minimum squared error)
        # In Auto-Encoders the self.input is the output, we just use the hidden layer to learn interesting representation
        # or weights for latent factors
        cost = (1/2*self.m) * pow(np.sum(self.outLayerState - self.input), 2) + \
               ((self.lambda_/2) * reg) + \
               (self.beta * np.sum(divergence))
        
        print ('cost: \n', cost)
        

    def backwardProp(self):
        # Backward Propagation The derivative term for divergence, The below is just the derivative term repeated
        delta_rho = self.deltaSparsity(numInputs = self.m)

        # Compute the gradients for the Output state and hidden state
        delta_outLayer = -(self.input - self.outLayerState)
        logging.info('The shape of delta_outLayer is: %s', delta_outLayer.shape)
        delta_hidLayer = (np.dot(delta_outLayer, np.transpose(self.hid_to_out_wghts)) + \
                          + self.beta * delta_rho
                          ) * sigmoid(self.hidLayerState) * (1 - sigmoid(self.hidLayerState))
        logging.info('The shape of delta_hidLayer is: %s', delta_hidLayer.shape)

        # Compute the weight gradients:
        delta_hid_to_out_wghts = (1 / self.m) * np.dot(np.transpose(self.hidLayerState),
                                                  delta_outLayer) + (self.lambda_ * self.hid_to_out_wghts)
        logging.info('The shape of delta_self.hid_to_out_wghts is: %s', delta_hid_to_out_wghts.shape)

        delta_inp_to_hid_wghts = (1 / self.m) * np.dot(np.transpose(self.input),
                                                  delta_hidLayer) + (self.lambda_ * self.inp_to_hid_wghts)
        logging.info('The shape of delta_self.inp_to_hid_wghts is: %s', delta_inp_to_hid_wghts.shape)
        
        
        # Compute the New Weights:
        
        
        
    def computeNetwork(self, input):
        self.setData(input)
        self.initializeParams()
        self.forwardProp(activationType="sigmoid")
        self.backwardProp()
        
debug = True

if debug:
    b = SparseAutoEncoders(numInpUnits = 3,
                       numHidUnits = 2,
                       numOutUnits = 3,
                       rho = 0.05, lambda_=0.5, beta =0.3).computeNetwork(input=np.array([[1, 2, 3]], dtype=float))
#
#     / Users / sam / App - Setup / anaconda / bin / python3
#     .6 / Users / sam / All - Program / App / Deep - Neural - Nets / AutoEncoders / Tools.py
#     The
#     self.input is: [[1.  2.  3.]]
#     divergence
#     for each hidden units:
#         [0.12638137 - 0.03801374]
#     cost:
#     20.9276832596
# rho_delta:
# [[1.5898132   0.76927993]]
#
# Process
# finished
# with exit code 0

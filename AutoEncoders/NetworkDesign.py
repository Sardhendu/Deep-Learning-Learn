import numpy as np
import math
from collections import OrderedDict

class NetworkDesign():
    def __init__(self, numInpUnits, numHidUnits, numOutUnits):
        """
        :param numInpUnits: Number of hidden units
        :param numHidUnits: Number of Hidden activation units
        :param numOutUnits: Number of output units,
                            For an autoencoder this has to be equal
                            to the nunImpUnits
        :param rho: The sparse parameter value
        :param lamda: The weight decay parameter value
        :param beta:
        """
        self.numInpUnits = numInpUnits
        self.numHidUnits = numHidUnits
        self.numOutUnits = numOutUnits
        self.params = OrderedDict()
        
    def initBias(self):
        stackedBias = []
        stackedBias.append(np.zeros((1,self.numHidUnits)))
        stackedBias.append(np.zeros((1,self.numOutUnits)))
        self.params['b'] = stackedBias
    
    def initWeights(self, distribution='uniform', seed=565):
        stackedWeights = []
        range = np.sqrt(6) / np.sqrt(self.numInpUnits + self.numInpUnits + 1)
        randState = np.random.RandomState(seed)
        
        if distribution=='uniform':
            stackedWeights.append(np.asarray(
                    randState.uniform(low=-range, high=range, size=(self.numInpUnits, self.numHidUnits))
            ))
            stackedWeights.append(np.asarray(
                    randState.uniform(low=-range, high=range, size=(self.numHidUnits, self.numOutUnits))
            ))

        self.params['w'] = stackedWeights


#
# a = NetworkDesign(numInpUnits=64,
#                   numHidUnits=25,
#                   numOutUnits=64,
#                   rho=0.05, lamda=0.2, beta=0.3)
#
# a.initBias
# a.initWeights

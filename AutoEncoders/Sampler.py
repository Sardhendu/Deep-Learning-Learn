import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import StratifiedKFold


class Sampler():
    def __init__(self, seed=3251):
        self.random_state = seed
    
    # Stratified Sampling:
    def getStratifiedSamples(self, dataIN, targetIN, testSize=0.2, get_indices=False):
        if len(dataIN) != len(targetIN):
            raise ValueError('The Dataset and Target should be of equal length')
        
        sss = StratifiedShuffleSplit(n_splits=1, test_size=testSize, random_state=self.random_state)
        for trainIndex, testIndex in sss.split(dataIN, targetIN):
            trainX = dataIN[trainIndex]
            trainY = targetIN[trainIndex]
            testX = dataIN[testIndex]
            testY = targetIN[testIndex]
        
        if get_indices:
            traintest_Indices = OrderedDict
            traintest_Indices['trainIndex'] = trainIndex
            traintest_Indices['testIndex'] = testIndex
            return trainX, trainY, testX, testY, traintest_Indices
        else:
            return trainX, trainY, testX, testY
    
    
    # Stratified Samling generate Multiple Spits
    def getStratifiedSplits(self, dataIN, targetIN, testSize=0.2, numSplits=3, get_indices=False):
        if len(dataIN) != len(targetIN):
            raise ValueError('The Dataset and Target should be of equal length')
        
        sss = StratifiedShuffleSplit(n_splits=numSplits, test_size=testSize, random_state=self.random_state)
        for trainIndex, testIndex in sss.split(dataIN, targetIN):
            
            trainX = dataIN[trainIndex]
            trainY = targetIN[trainIndex]
            testX = dataIN[testIndex]
            testY = targetIN[testIndex]
            
            if get_indices:
                traintest_Indices = OrderedDict
                traintest_Indices['trainIndex'] = trainIndex
                traintest_Indices['testIndex'] = testIndex
                yield trainX, trainY, testX, testY, traintest_Indices
            else:
                yield trainX, trainY, testX, testY
    
    
    # Generate k-Foldbatches
    def getStratifiedBatches(self, dataIN, targetIN, numBatches=3, get_indices=False):
        if len(dataIN) != len(targetIN):
            raise ValueError('The Dataset and Target should be of equal length')
        
        print(dataIN.shape, targetIN.shape)
        skf = StratifiedKFold(n_splits=numBatches, random_state=self.random_state)
        for trainIndex, testIndex in skf.split(dataIN, targetIN):
            
            trainX = dataIN[trainIndex]
            trainY = targetIN[trainIndex]
            testX = dataIN[testIndex]
            testY = targetIN[testIndex]
            
            if get_indices:
                traintest_Indices = OrderedDict
                traintest_Indices['trainIndex'] = trainIndex
                traintest_Indices['testIndex'] = testIndex
                yield trainX, trainY, testX, testY, traintest_Indices
            else:
                yield trainX, trainY, testX, testY
                
                
                
       
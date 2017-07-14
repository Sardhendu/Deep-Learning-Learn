from __future__ import division
from __future__ import print_function


import os, glob, sys


import pickle
import random


from collections import OrderedDict

# import imutils
import numpy as np
from skimage.io import imread

cifar10ParentPath = "/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/CIFAR10"



def CIFAR10():
    labelDict = OrderedDict()
    directories =  [dir for dir in os.listdir(cifar10ParentPath) if dir != ".DS_Store"]

    # Build a big pickle file with all the png file converted in matrix form
    for num, folderName in enumerate(directories):
        labelDict[num+1] = folderName
        folderPath = os.path.join(cifar10ParentPath, folderName)

        pngFiles = os.listdir(folderPath)
        imagePathArr = [os.path.join(folderPath, imgPath) for imgPath in pngFiles]

        dataMatrix = np.array([imread(image) for image in imagePathArr])
        labelMatrix = np.tile(num + 1, len(dataMatrix)).reshape(-1, 1)
        
        if num==0:
            dataBulkMatrix = dataMatrix
            labelBulkMatrix = labelMatrix
        else:
            dataBulkMatrix = np.vstack((dataBulkMatrix, dataMatrix))
            labelBulkMatrix = np.vstack((labelBulkMatrix, labelMatrix))
        
        # if num ==3:
        #     break
    print (dataBulkMatrix.shape, labelBulkMatrix.shape)

    with open(os.path.join(cifar10ParentPath, "dataFull") + '.pickle', 'wb') as f:
        pickleData = {
            'trainingData': dataBulkMatrix,
            'trainingLabels': labelBulkMatrix,
            'labelDict': labelDict
        }
        pickle.dump(pickleData, f, pickle.HIGHEST_PROTOCOL)


debugg = False
if debugg:
    CIFAR10()
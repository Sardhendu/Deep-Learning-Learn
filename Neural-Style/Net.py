from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import numpy as np
import scipy.io


parser.add_argument('--model_weights', type=str,
    default='imagenet-vgg-verydeep-19.mat',
    help='Weights and biases of the VGG-19 network.')
args = parser.parse_args()
vgg_rawnet     = scipy.io.loadmat(args.model_weights)
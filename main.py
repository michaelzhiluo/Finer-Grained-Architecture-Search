import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.python import pywrap_tensorflow
import os
import logging
import sys
from GeneralizedConv import GeneralizedConvNetwork
from train import DARTSAlgorithm
from DataSet import DataSet
from MNISTConfig import mnist_config
from Cifar10Config import cifar10_config

config = cifar10_config
sess = tf.Session()
network = GeneralizedConvNetwork(config)
dataset = DataSet(config)
init = tf.global_variables_initializer()
sess.run(init)
tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
tf_config.gpu_options.allow_growth = True

trainer = DARTSAlgorithm(sess, network, dataset, config)
trainer.train()







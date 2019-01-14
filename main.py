import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import seaborn as sns
from tensorflow.python.tools import inspect_checkpoint as chkp
from IPython import embed
from tensorflow.python import pywrap_tensorflow
import os
import logging
import sys
from GeneralizedConv import GeneralizedConvNetwork
from train import DARTSAlgorithm
from DataSet import DataSet


config = {
	"meta_iterations": 30000,
	"train_batch_size": 256,
	"weight_lr": 0.001,
	"meta_lr": 0.001,
	"beta": 0.003,
	"dataset": "mnist",
	"model":
	{
		"layers": [

			{
				"type": "Dense",
				"filter_height": 5,
				"filter_width": 5,
				"in_channels": 1,
				"out_channels": 32,
				"output_dim": [14, 14], # Should be a perfect square
				"weight_initializer": tf.contrib.layers.xavier_initializer_conv2d(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				
				#"pool": 
				#	{
				#		"pool_type": "avg", # max/avg
				#		"kernel_size": 2,
				#		"stride": 2,
				#	},

			},

			{
				"type": "FC",
				"input": 14*14*32,
				"output": 1024,
				"weight_initializer": tf.contrib.layers.xavier_initializer(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				#"norm": "batch",
			},
			
			{
				"type": "FC",
				"input": 1024,
				"output": 10,
				"weight_initializer": tf.contrib.layers.xavier_initializer(),
				"bias_initalizer":  tf.zeros_initializer(),
				#"activation": "relu",
				#"norm": "batch",
			},

		]
	},
}
'''
			{
				"type": "Conv",
				"filter_height": 5,
				"filter_width": 5,
				"in_channels": 1,
				"out_channels": 32,
				"stride": 1,
				"weight_initializer": tf.contrib.layers.xavier_initializer_conv2d(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				"pool": 
					{
						"pool_type": "avg", # max/avg
						"kernel_size": 2,
						"stride": 2,
					},
				#"norm": "batch", #layer or bath
			},
			'''
sess = tf.Session()
network = GeneralizedConvNetwork(config)
dataset = DataSet(config)
init = tf.global_variables_initializer()
sess.run(init)
tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1)
tf_config.gpu_options.allow_growth = True

trainer = DARTSAlgorithm(sess, network, dataset, config)
trainer.train()







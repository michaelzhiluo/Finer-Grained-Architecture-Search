import numpy as np
import tensorflow as tf


class ConnectionLayer(object):

	def __init__(self, sess, input, num_outputs):
		self.input = input 
		with tf.variable_scope("Hyperparameters"):
			alpha = tf.Variable(, name = "alpha")
	
	def update_weights(self):
		return

	def update_hyper(self):
		return

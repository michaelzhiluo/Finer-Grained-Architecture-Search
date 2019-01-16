import numpy as np
import tensorflow as tf
from categorical import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from tensorflow.examples.tutorials.mnist import input_data
import random
from functools import reduce
from tensorflow.contrib.layers.python import layers as tf_layers

def FCLayer(inp, W, b, activation_fn, norm_type):
	input_dim = inp.get_shape()
	assert len(input_dim)>1, "Input Dimension must be > 1"

	if(len(input_dim)>2):
		inp = tf.reshape(inp, [-1, reduce(lambda i, j: i*j, input_dim[1:])])

	assert inp.get_shape()[1]==W.get_shape()[0], "Input and FC Weight do not match"

	return Normalize(tf.add(tf.matmul(inp, W), b), norm_type, activation_fn, False)

def ConvLayer(inp, weight, bias, strides, activation_fn, norm_type, pool_config):
	input_dim = inp.get_shape()
	assert len(input_dim)==4, "Input Dimension must be in NHWC format"
	assert input_dim[3]==weight.get_shape()[2], "Input and Weight must have same # of input channels"

	conv_output = Normalize(tf.nn.conv2d(inp, weight, strides = [1, strides, strides, 1], padding = 'SAME') + bias, norm_type, activation_fn, False)

	pool_output = Pool(conv_output, pool_config["pool_type"], ksize = pool_config["kernel_size"], strides =  pool_config["stride"]) if pool_config else conv_output

	return pool_output

def Pool(inp, pool_type, ksize, strides):
	input_dim = inp.get_shape()
	assert len(input_dim)==4, "Input Dimension must be in NHWC format"

	if pool_type == "max":
		return tf.nn.max_pool(inp, ksize = [1,ksize, ksize, 1], strides = [1, strides, strides, 1], padding = 'SAME')
	elif pool_type =="avg":
		return tf.nn.avg_pool(inp, ksize = [1,ksize, ksize, 1], strides = [1, strides, strides, 1]		, padding = 'SAME')

def Normalize(inp, norm_type, activation, reuse, scope=None):
	if norm_type == 'batch':
		return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope = scope)
	elif norm_type == 'layer':
		return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope = scope)
	elif norm_type is None:
		if activation is not None:
			return activation(inp)
	return inp

def get_activation_function(name):
	if name == "relu":
		return tf.nn.relu
	elif name == "tanh":
		return tf.nn.tanh
	elif name == "sigmoid":
		return tf.nn.sigmoid
	return tf.identity

def DenseLayer(inputs, 
			output_dim, 
			weight,
			bias,
			weight_train,
			exploration,
			activation_fn, 
			norm_type, 
			pool_config,
			hyperparam_scope = "Hyperparameters",
			reuse = True):
	'''
	DenseLayer takes any image, converts it to [batchsize, image_size]. Image size is all the image dimensions multiplied together.
	For example, a 100 10x10x3 standard images will reduce [100, 10*10*3].

	inputs-Input tensor of shape [batch_size] + [Any Image Shape] (it will be flattened anyways)
	output- Scalar Output Size e.g 100
	num_filters - Output filters 
	'''

	#Flatten/vectorizes input to [batch_size, input_size]
	inputs = tf.contrib.layers.flatten(inputs)

	#Weight reshaped to [num_filter, weights per filter]
	weight_shape = weight.get_shape()
	weight = tf.reshape(weight, [weight_shape[3], weight_shape[0]*weight_shape[1]*weight_shape[2]])

	with tf.variable_scope(hyperparam_scope, reuse = reuse):
		alpha = tf.get_variable("alpha" + str(output_dim[0]), shape=[output_dim[0]*output_dim[1], weight.get_shape()[1], inputs.get_shape()[1].value])
		print(alpha)
	#SoftMax Alpha
	s_alpha = tf.nn.softmax(alpha, name="softmax_alpha")

	# If exploration, sample uniformly (infinitely high temperature), elsewise sample from your learned alphas with temperature set to 1
	dist = tf.cond(exploration > 0, lambda: gumbel_softmax(alpha, 100000000000, True), lambda:  gumbel_softmax(alpha, 1, True))

	sampled_connections = tf.einsum("abc,dc->dab", dist, inputs)

	max_connections = tf.gather(inputs, tf.argmax(s_alpha, axis = 2, output_type=tf.int32, name="argmax_alpha"), axis=1)

	connections = tf.cond(weight_train>0, lambda: sampled_connections, lambda: sampled_connections)

	multiplied_output = tf.einsum("aij,bj->aib", connections, weight)

	dense_output = tf.reshape(tf.nn.bias_add(multiplied_output, bias), [-1, output_dim[0], output_dim[1], weight_shape[3]])

	output = Normalize(dense_output, norm_type, activation_fn, False)


	pool_output = Pool(output, pool_config["pool_type"], ksize = pool_config["kernel_size"], strides =  pool_config["stride"]) if pool_config else dense_output

	return pool_output


class Explore(object):
	def __init__(self, initial_exp, exp_decay):
		self.cur_exp = initial_exp
		self.exp_decay  = exp_decay

	def explore(self):
		self.cur_exp *= self.exp_decay
		return 1 if random.random() <= self.cur_exp else 0


import numpy as np
import tensorflow as tf
from categorical import sample_gumbel, gumbel_softmax_sample, gumbel_softmax
from tensorflow.examples.tutorials.mnist import input_data

def DenseLayer(inputs, 
			output, 
			num_weights_per_filter, 
			num_filters,
			weight_train,
			exploration,
			activation_fn = tf.nn.relu,
			weights_initalizer = tf.initializers.random_uniform(0, 1),#tf.contrib.layers.xavier_initializer(),
			biases_initializer = tf.zeros_initializer(),
			weight_scope = "Weights",
			hyperparam_scope = "Hyperparameters"):
	'''
	DenseLayer takes any image, converts it to [batchsize, image_size]. Image size is all the image dimensions multiplied together.
	For example, a 100 10x10x3 standard images will reduce [100, 10*10*3].

	inputs-Input tensor of shape [batch_size] + [Any Image Shape] (it will be flattened anyways)
	output- Scalar Output Size e.g 100
	num_filters - Output filters 
	'''

	#Flatten/vectorizes input to [batch_size, input_size]
	inputs = tf.contrib.layers.flatten(inputs)

	#Declaring tf weight variable of size output_f
	with tf.variable_scope(weight_scope, reuse = tf.AUTO_REUSE):
		weights = tf.Variable(weights_initalizer([num_filters, num_weights_per_filter])) #1]) # Since image is flattened, it's only one channel
		bias = tf.Variable(biases_initializer([num_filters]))

	with tf.variable_scope(hyperparam_scope, reuse = tf.AUTO_REUSE):
		alpha = tf.get_variable("hyperparam", shape=[output, num_weights_per_filter, inputs.get_shape()[1].value])

	#SoftMax Alpha
	s_alpha = tf.nn.softmax(alpha)
	# If exploration, sample unifromly, else wise sample from your learned alphas
	#dist = tf.cond(exploration>0, lambda: gumbel_softmax( , 0.5, True), lambda: gumbel_softmax(s_alpha, 0.5, True))

	dist = gumbel_softmax(s_alpha, 0.5, True)
	
	sampled_connections = tf.einsum("abc,dc->dab", dist, inputs)

	max_connections = tf.gather(inputs, tf.argmax(s_alpha, axis = 2, output_type=tf.int32), axis=1)

	connections = tf.cond(weight_train>0, lambda: max_connections, lambda: sampled_connections)

	multiplied_output = tf.einsum("aij,bj->aib", connections, weights)

	final_output = activation_fn(tf.nn.bias_add(multiplied_output, bias))
	return final_output

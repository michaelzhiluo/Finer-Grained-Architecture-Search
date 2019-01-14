import tensorflow as tf
from GeneralizedConv import GeneralizedConvNetwork
from utils import *
from DataSet import DataSet

# Performs DARTS Bilevel optimization
class DARTSAlgorithm(object):

	def __init__(self, sess, network, dataset, config):
		self.sess = sess
		self.network = network
		self.dataset = dataset
		self.config = config
		self.iterations = self.config["meta_iterations"]
		#if "explore" in config and self.config["explore"]:
			#self.explore = Explore()

	# One iteration of training weights
	def train_weights(self):
		train_input, train_label = self.dataset.batch("train", self.config["train_batch_size"])

		opt = self.sess.run(self.network.weight_update, feed_dict={self.network.train_input: train_input, self.network.train_label: train_label, self.network.train_weights: 1, self.network.exploration: 0})

	# One iteration of training hyper parameters
	def train_hypers(self):
		train_input, train_label = self.dataset.batch("train", self.config["train_batch_size"])
		test_input, test_label = self.dataset.batch("test", self.config["train_batch_size"])

		opt = self.sess.run([self.network.hyper_update],feed_dict={
			self.network.train_input: train_input,
			self.network.train_label: train_label,
			self.network.test_input: test_input,
			self.network.test_label: test_label,
			self.network.exploration: 0,
			self.network.train_weights: 0
			})

	def accuracy(self, dataset_type):
		inp, label = self.dataset.batch(dataset_type, self.config["train_batch_size"])
		loss, acc = self.sess.run([self.network.train_loss, self.network.weight_accuracy], 
				feed_dict={self.network.train_input: inp, self.network.train_label: label, self.network.train_weights: 1, self.network.exploration: 0})
		return loss, acc


	def train(self):
		for meta_step in range(0, self.iterations):
			print("----------------------------META-ITERATION " + str(meta_step) + "----------------------------")
			print("SWITCHING TO WEIGHT OPT")
			self.train_weights()
			loss, acc = self.accuracy("train")
			print("Post-Weight Update with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))

			print("SWITCHING TO HYPERPARAM OPT")
			self.train_hypers()
			loss, acc = self.accuracy("test")
			print("Post-Hyper Update with normal batch, Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " +  "{:.5f}".format(acc))

			loss, acc = self.accuracy("validation")
			print("Validation Accuracy, Minibatch Loss= " + "{:.6f}".format(loss) + ", Accuracy= " +  "{:.5f}".format(acc))






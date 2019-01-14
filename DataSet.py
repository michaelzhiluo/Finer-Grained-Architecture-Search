import numpy as np
from Cifar10 import get_cifar_10
from tensorflow.examples.tutorials.mnist import input_data


class DataSet(object):
	'''
		Training Data - Data used to train weights
		Test Data - Data used to train hyperparameters / alphas
		Validation Data - Data not used for any training but for validation accuracy
	'''
	def __init__(self, config):
		self.batch_size = config["train_batch_size"]
		self.config = config
		self.data = {"train": None, "test": None, "validation": None}
		self.index = {"train": 0, "test": 0, "validation": 0}

		if self.config["dataset"]=="cifar10": 
			train_data, train_label = get_cifar_10("train")
			self.data["train"] = (train_data, train_label)

			test_data, test_label = get_cifar_10("test")
			test_data, test_label = self.shuffle_data(test_data, test_label)
			self.data["test"] = (test_data[:5000], test_label[:5000])
			self.data["validation"] = (test_data[5000:], test_label[5000:])
		elif self.config["dataset"]=="mnist":
			mnist = input_data.read_data_sets("/data/mluo/tmp/data/", one_hot=True)
			self.data["train"] = (mnist.train.images, mnist.train.labels)
			self.data["test"] = (mnist.test.images, mnist.test.labels)
			self.data["validation"] = (mnist.validation.images, mnist.validation.labels)

	def batch(self, dataset_type, batch_size):
		dataset_length = len(self.data[dataset_type][0])
		if self.index[dataset_type] + batch_size > dataset_length:
			#Images and Labels before the end of Epoch
			images1 = self.data[dataset_type][0][self.index[dataset_type]:dataset_length]
			labels1 = self.data[dataset_type][1][self.index[dataset_type]:dataset_length]

			#Shuffle Data after Epoch is complete
			a,b = self.shuffle_data(self.data[dataset_type][0], self.data[dataset_type][1])
			self.data[dataset_type] = (a,b)

        	#Images and Labels in the new Epoch
			batch_size -= dataset_length - self.index[dataset_type]
			self.index[dataset_type] = batch_size
			images2 = self.data[dataset_type][0][:self.index[dataset_type]]
			labels2 =self.data[dataset_type][1][:self.index[dataset_type]]

			return np.concatenate((images1, images2), axis = 0), np.concatenate((labels1, labels2), axis = 0)
		
		start = self.index[dataset_type]
		self.index[dataset_type] += batch_size
		return self.data[dataset_type][0][start: self.index[dataset_type]], self.data[dataset_type][1][start: self.index[dataset_type]]

	def shuffle_data(self, data, label):
		perm = np.arange(len(data))
		np.random.shuffle(perm)
		return data[perm], label[perm]



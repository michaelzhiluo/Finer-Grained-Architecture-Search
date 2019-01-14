import tensorflow as tf

cifar10_config = {
	"meta_iterations": 100000,
	"train_batch_size": 256,
	"weight_lr": 0.001,
	"meta_lr": 0.001,
	"beta": 0.0005,
	"dataset": "cifar10",
	"model":
	{
		"layers": [

			{
				"type": "Dense",
				"filter_height": 5,
				"filter_width": 5,
				"in_channels": 3,
				"out_channels": 64,
				"output_dim": [16, 16],
				"weight_initializer": tf.contrib.layers.xavier_initializer_conv2d(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				#"norm": "batch", #layer or bath
			},

			{
				"type": "Conv",
				"filter_height": 5,
				"filter_width": 5,
				"in_channels": 64,
				"out_channels": 64,
				"stride": 1,
				"weight_initializer": tf.contrib.layers.xavier_initializer_conv2d(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				"pool": 
					{
						"pool_type": "max", # max/avg
						"kernel_size": 3,
						"stride": 2,
					},
				#"norm": "batch", #layer or bath
			},

			{
				"type": "FC",
				"input": 8*8*64,
				"output": 384,
				"weight_initializer": tf.contrib.layers.xavier_initializer(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				#"norm": "batch",
			},

			{
				"type": "FC",
				"input": 384,
				"output": 192,
				"weight_initializer": tf.contrib.layers.xavier_initializer(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				#"norm": "batch",
			},
			
			{
				"type": "FC",
				"input": 192,
				"output": 10,
				"weight_initializer": tf.contrib.layers.xavier_initializer(),
				"bias_initalizer":  tf.zeros_initializer(),
				#"activation": "relu",
				#"norm": "batch",
			},

		]
	},
}




cifar10_conv_config = {
	"meta_iterations": 100000,
	"train_batch_size": 256,
	"weight_lr": 0.001,
	"meta_lr": 0.001,
	"beta": 0.003,
	"dataset": "cifar10",
	"model":
	{
		"layers": [

			{
				"type": "Conv",
				"filter_height": 5,
				"filter_width": 5,
				"in_channels": 3,
				"out_channels": 64,
				"stride": 1,
				"weight_initializer": tf.contrib.layers.xavier_initializer_conv2d(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				"pool": 
					{
						"pool_type": "max", # max/avg
						"kernel_size": 3,
						"stride": 2,
					},
				#"norm": "batch", #layer or bath
			},

			{
				"type": "Conv",
				"filter_height": 5,
				"filter_width": 5,
				"in_channels": 64,
				"out_channels": 64,
				"stride": 1,
				"weight_initializer": tf.contrib.layers.xavier_initializer_conv2d(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				"pool": 
					{
						"pool_type": "max", # max/avg
						"kernel_size": 3,
						"stride": 2,
					},
				#"norm": "batch", #layer or bath
			},

			{
				"type": "FC",
				"input": 8*8*64,
				"output": 384,
				"weight_initializer": tf.contrib.layers.xavier_initializer(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				#"norm": "batch",
			},

			{
				"type": "FC",
				"input": 384,
				"output": 192,
				"weight_initializer": tf.contrib.layers.xavier_initializer(),
				"bias_initalizer":  tf.zeros_initializer(),
				"activation": "relu",
				#"norm": "batch",
			},
			
			{
				"type": "FC",
				"input": 192,
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
'''
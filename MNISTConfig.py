import tensorflow as tf

mnist_config = {
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
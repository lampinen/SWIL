from __future__ import print_function
from __future__ import division 

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

###### configuration ###########################################################

config = {
    "num_runs": 10,
    "layer_sizes": [128, 64, 32, 64, 128]
}

###### MNIST data loading and manipulation #####################################
# downloaded from https://pjreddie.com/projects/mnist-in-csv/

train_data = np.loadtxt("MNIST_data/mnist_train.csv", delimiter = ",")
test_data = np.loadtxt("MNIST_data/mnist_test.csv", delimiter = ",")

def process_data(raw_dataset):
    """Get data split into dict with labels and images"""
    labels = dataset[:, 0]
    images = dataset[:, 1:]/255.
    data = {"labels": labels, "images": images}
    return data

train_data = process_data(train_data)
test_data = process_data(test_data)

###### Build model func ########################################################

class MNIST_autoenc(object):
    """MNIST autoencoder architecture, with or without replay buffer"""

    def __init__(self, replay_type, layer_sizes):
        """Create a MNIST_autoenc model. 
            replay_type: one of ["None", "Random", "SWIL"]
            layer_sizes: list of the hidden layer sizes of the model
        """
        self.input_ph = tf.placeholder(tf.float32, [None, 784])
        self.lr_ph = tf.placeholder(tf.float32)

        min_layer_size = min(layer_sizes)
        net = self.input_ph
        for h_size in layer_sizes:
            net = slim.layers.fully_connected(net, h_size, activation_fn=tf.nn.relu)
            if h_size == min_layer_size: 
                self.bottleneck_rep = net
        self.output = slim.layers.fully_connected(net, 784, activation_fn=tf.nn.sigmoid)
        self.loss = tf.nn.l2_loss(self.output-self.input_ph)


    def base_train(self, dataset):
        """Train the model on a dataset"""
        pass
    
    def new_data_train(self, new_dataset, old_dataset):
        """Assuming the model has been trained on old_dataset, add new_dataset,
        possibly interleaving from old"""
        pass

    def eval(self, dataset):
        """Evaluates model on the given dataset"""
        pass

###### Run stuff ###############################################################

for left_out_class in range(10):
    for replay_type in ["None", "Random", "SWIL"]:
        for run in range(config["num_runs"]):
            filename_prefix = "m_%s_lo%i_run%i_" %(replay_type,
                                                   left_out_class,
                                                   run)
            np.random.seed(run)
            tf.set_random_seed(run)


            model = MNIST_autoenc(replay_type=replay_type,
                                  layer_sizes=config["layer_sizes")



from __future__ import print_function
from __future__ import division 

from collections import Counter

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import matplotlib.pyplot as plot

###### configuration ###########################################################

config = {
    "num_runs": 10,
    "batch_size": 10,
    "base_learning_rate": 0.001,
    "base_lr_decay": 0.9,
    "base_lr_decays_every": 1,
    "base_lr_min": 0.0001,
    "new_learning_rate": 0.0001,
    "new_lr_decay": 0.9,
    "new_lr_decays_every": 1,
    "new_lr_min": 5e-6,
    "base_training_epochs": 20,
    "new_training_epochs": 20,
    "new_batch_num_replay": 5, # how many of batch of new items are replays
                                # if replay is on
    "SW_by": "reps", # one of "images" or "reps", what feature space to do
		       # the similarity weighting in
    "softmax_temp": 1, # temperature for SWIL replay softmax
    "output_path": "./results/",
    "layer_sizes": [128, 32, 16, 32, 128]
}

###### MNIST data loading and manipulation #####################################
# downloaded from https://pjreddie.com/projects/mnist-in-csv/

train_data = np.loadtxt("MNIST_data/mnist_train.csv", delimiter = ",")
test_data = np.loadtxt("MNIST_data/mnist_test.csv", delimiter = ",")

def process_data(dataset):
    """Get data split into dict with labels and images"""
    labels = dataset[:, 0]
    images = dataset[:, 1:]/255.
    data = {"labels": labels, "images": images}
    return data

train_data = process_data(train_data)
test_data = process_data(test_data)

###### Build model func ########################################################

def softmax(x, T=1):
    """Compute the softmax function at temperature T"""
    if T != 1:
        x /= T
    x -= np.amax(x)
    x = np.exp(x)
    x /= np.sum(x)
    return x

def to_unit_rows(x):
    """Converts row vectors of a matrix to unit vectors"""
    return x/np.expand_dims(np.sqrt(np.sum(x**2, axis=1)), -1)

def _display_image(x):
    x = np.reshape(x, [28, 28])
    plot.figure()
    plot.imshow(x, vmin=0, vmax=1)

class MNIST_autoenc(object):
    """MNIST autoencoder architecture, with or without replay buffer"""

    def __init__(self, replay_type, layer_sizes):
        """Create a MNIST_autoenc model. 
            replay_type: one of ["None", "Random", "SWIL"]
            layer_sizes: list of the hidden layer sizes of the model
        """
        self.replay_type = replay_type
        self.base_lr = config["base_learning_rate"]
        self.new_lr = config["new_learning_rate"]

        self.input_ph = tf.placeholder(tf.float32, [None, 784])
        self.lr_ph = tf.placeholder(tf.float32)

        self.bottleneck_size = min(layer_sizes)
        net = self.input_ph
        for h_size in layer_sizes:
            net = slim.layers.fully_connected(net, h_size, activation_fn=tf.nn.relu)
            if h_size == self.bottleneck_size: 
                self.bottleneck_rep = net
        self.output = slim.layers.fully_connected(net, 784, activation_fn=tf.nn.sigmoid)
        self.loss = tf.nn.l2_loss(self.output-self.input_ph)

        self.optimizer = tf.train.GradientDescentOptimizer(self.lr_ph)
        self.train = self.optimizer.minimize(tf.reduce_mean(self.loss))

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def base_train(self, dataset, nepochs=100, log_file_prefix=None, test_dataset=None):
        """Train the model on a dataset"""
        if log_file_prefix is not None:
            if test_dataset is not None:
                with open(config["output_path"] + log_file_prefix + "base_test_losses.csv", "w") as fout:
                    fout.write("epoch, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n" % tuple(range(10)))
		    losses = self.eval(test_dataset)
		    fout.write(("0, ") + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))
	    with open(config["output_path"] + log_file_prefix + "base_train_losses.csv", "w") as fout:
		fout.write("epoch, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n" % tuple(range(10)))
		losses = self.eval(dataset)
		fout.write(("0, ") + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))

        batch_size = config["batch_size"]
        for epoch in range(1, nepochs + 1):
            order = np.random.permutation(len(dataset["labels"]))
            for batch_i in xrange(len(dataset["labels"])//batch_size):
                this_batch_images = dataset["images"][order[batch_i*batch_size:(batch_i+1)*batch_size], :]
                self.sess.run(self.train, feed_dict={
                        self.input_ph: this_batch_images,
                        self.lr_ph: self.base_lr 
                    })

            # eval
            if log_file_prefix is not None:
                if test_dataset is not None:
                    with open(config["output_path"] + log_file_prefix + "base_test_losses.csv", "a") as fout:
                        losses = self.eval(test_dataset)
			fout.write(("%i, " % epoch) + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))
		with open(config["output_path"] + log_file_prefix + "base_train_losses.csv", "a") as fout:
		    losses = self.eval(dataset)
		    fout.write(("%i, " % epoch) + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))

	    # update lr
	    if epoch > 0 and epoch % config["base_lr_decays_every"] == 0 and self.base_lr > config["base_lr_min"]: 
		self.base_lr *= config["base_lr_decay"]
         
    def new_data_train(self, new_dataset, old_dataset=None, nepochs=100, test_dataset=None, log_file_prefix=None):
        """Assuming the model has been trained on old_dataset, tune on
        new_dataset, possibly interleaving from old"""
        softmax_temp = config["softmax_temp"]
        batch_size = config["batch_size"]
        if self.replay_type != "None":
            old_batch_size = config["new_batch_num_replay"]
            new_batch_size = batch_size - old_batch_size 
        
        if log_file_prefix is not None:
            if self.replay_type != "None": 
                with open(config["output_path"] + log_file_prefix + "replay_labels_encountered.csv", "w") as fout:
                    fout.write("epoch, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n" % tuple(range(10)))
            if test_dataset is not None:
                with open(config["output_path"] + log_file_prefix + "new_test_losses.csv", "w") as fout:
                    fout.write("epoch, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n" % tuple(range(10)))
		    losses = self.eval(test_dataset)
		    fout.write(("0, ") + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))
	    with open(config["output_path"] + log_file_prefix + "new_train_losses.csv", "w") as fout:
		fout.write("epoch, %i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n" % tuple(range(10)))
		losses = self.eval(new_dataset)
		fout.write(("0, ") + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))

        for epoch in range(1, nepochs + 1):
            if self.replay_type != "None":
                replay_labels_encountered = Counter()

            order = np.random.permutation(len(new_dataset["labels"]))
            if self.replay_type == "Random":
                old_order = np.random.permutation(len(old_dataset["labels"]))
            elif self.replay_type == "SWIL":
		if config["SW_by"] == "reps":
		    old_dataset_reps = self.get_reps(old_dataset["images"])
		else:
		    old_dataset_reps = old_dataset["images"]
		# standardize
		old_dataset_reps_means = np.mean(old_dataset_reps, axis=0)
		old_dataset_reps_sds = np.std(old_dataset_reps, axis=0)
		old_dataset_reps = (old_dataset_reps - old_dataset_reps_means)/old_dataset_reps_sds
		# normalize
		old_dataset_reps = to_unit_rows(old_dataset_reps)

            for batch_i in xrange(len(new_dataset["labels"])//batch_size):
                if self.replay_type == "None":
                    this_batch_images = new_dataset["images"][order[batch_i*batch_size:(batch_i+1)*batch_size], :]
                elif self.replay_type == "Random":
                    this_batch_images = np.concatenate(
                        [new_dataset["images"][order[batch_i*new_batch_size:(batch_i+1)*new_batch_size], :],
                         old_dataset["images"][old_order[batch_i*old_batch_size:(batch_i+1)*old_batch_size], :]])
                    replay_labels_encountered.update(old_dataset["labels"][old_order[batch_i*old_batch_size:(batch_i+1)*old_batch_size]])
                else: # SWIL
                    this_batch_new = new_dataset["images"][order[batch_i*new_batch_size:(batch_i+1)*new_batch_size], :]
		    if config["SW_by"] == "reps":
			this_batch_new_reps = self.get_reps(this_batch_new)
		    else:
			this_batch_new_reps = this_batch_new
                    # standardize
                    this_batch_new_reps_reps = (this_batch_new_reps - old_dataset_reps_means)/old_dataset_reps_sds
                    # normalize
                    this_batch_new_reps = to_unit_rows(this_batch_new_reps)

                    dots = np.dot(old_dataset_reps, this_batch_new_reps.transpose())

                    probabilities = softmax(np.amax(dots, axis=1), T=softmax_temp) 
                
                    # note that this is actually pretending that sampling with
                    # replacement is sampling without, but on a dataset this 
                    # large that shouldn't really be an issue unless the temp
                    # is too small...
                    replay_samples = np.where(np.random.multinomial(old_batch_size, probabilities)) 
                    
                    this_batch_images = np.concatenate(
                        [this_batch_new,
                         old_dataset["images"][replay_samples]])

                    replay_labels_encountered.update(old_dataset["labels"][replay_samples])

                self.sess.run(self.train, feed_dict={
                        self.input_ph: this_batch_images,
                        self.lr_ph: self.new_lr 
                    })

	    # eval
	    if log_file_prefix is not None:
		if self.replay_type != "None": 
		    with open(config["output_path"] + log_file_prefix + "replay_labels_encountered.csv", "a") as fout:
			counts = [replay_labels_encountered[i] for i in range(10)] 
			fout.write(("%i, " % epoch) + "%i, %i, %i, %i, %i, %i, %i, %i, %i, %i\n" % tuple(counts))
		if test_dataset is not None:
		    with open(config["output_path"] + log_file_prefix + "new_test_losses.csv", "a") as fout:
			losses = self.eval(test_dataset)
			fout.write(("%i, " % epoch) + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))
		with open(config["output_path"] + log_file_prefix + "new_train_losses.csv", "a") as fout:
		    losses = self.eval(new_dataset)
		    fout.write(("%i, " % epoch) + "%f, %f, %f, %f, %f, %f, %f, %f, %f, %f\n" % tuple(losses))

	    # update lr
	    if epoch > 0 and epoch % config["new_lr_decays_every"] == 0 and self.new_lr > config["new_lr_min"]: 
		self.new_lr *= config["new_lr_decay"]




    def get_reps(self, images):
        """Gets bottleneck reps for the given images"""
        batch_size = config["batch_size"]
        reps = np.zeros([len(images), self.bottleneck_size])
        for batch_i in xrange((len(images)//batch_size) + 1):
            this_batch_images = images[batch_i*batch_size:(batch_i+1)*batch_size, :]
            reps[batch_i*batch_size:(batch_i+1)*batch_size, :] = self.sess.run(
                self.bottleneck_rep, feed_dict={
                    self.input_ph: this_batch_images 
                })
        return reps

    def get_loss(self, images):
        """Gets losses for the given images"""
        batch_size = config["batch_size"]
        loss = np.zeros([len(images)])
        for batch_i in xrange((len(images)//batch_size) + 1):
            this_batch_images = images[batch_i*batch_size:(batch_i+1)*batch_size, :]
            loss[batch_i*batch_size:(batch_i+1)*batch_size] = self.sess.run(
                self.loss, feed_dict={
                    self.input_ph: this_batch_images
                })
        return loss

    def eval(self, dataset):
        """Evaluates model on the given dataset. Returns list of losses where
        losses[i] is the average loss on digit i"""
        losses = self.get_loss(dataset["images"])
        losses_summarized = [np.sum(losses[dataset["labels"] == i])/np.sum(dataset["labels"] == i) for i in range(10)]
        return losses_summarized

    def display_output(self, image):
        """Runs an image and shows comparison"""
        output_image = self.sess.run(self.output, feed_dict={
                self.input_ph: np.expand_dims(image, 0) 
            })

        _display_image(image)
        _display_image(output_image)
        plot.show()



###### Run stuff ###############################################################

for run in range(config["num_runs"]):
    for left_out_class in range(10): 
	for replay_type in ["SWIL"]:#, "Random", "None"]:
	    for temperature in [0.1, 0.5, 1, 2, 10]:
		if temperature != 1 and replay_type != "SWIL":
		    continue 
		config["softmax_temp"] = temperature # ugly
		filename_prefix = "run%i_lo%i_m_%s_" %(run,
						       left_out_class,
						       replay_type)
		if replay_type == "SWIL":
		    filename_prefix += "swby_%s_t_%.1f_" % (config["SW_by"], config["softmax_temp"]) 
		print(filename_prefix)
		np.random.seed(run)
		tf.set_random_seed(run)


		model = MNIST_autoenc(replay_type=replay_type,
				      layer_sizes=config["layer_sizes"])

		indices = train_data["labels"] == left_out_class
		this_base_data = {"labels": train_data["labels"][np.logical_not(indices)],
				  "images": train_data["images"][np.logical_not(indices)]}
		#print(Counter(this_base_data["labels"]).most_common())
		this_new_data = {"labels": train_data["labels"][indices],
				 "images": train_data["images"][indices]}
		#print(Counter(this_new_data["labels"]).most_common())
		#model.display_output(this_base_data["images"][0])
		#model.display_output(this_new_data["images"][0])
		model.base_train(this_base_data, config["base_training_epochs"],
				 log_file_prefix=filename_prefix, test_dataset=test_data)
		#model.display_output(this_base_data["images"][0])
		#model.display_output(this_new_data["images"][0])
		model.new_data_train(this_new_data, this_base_data, config["new_training_epochs"],
				     log_file_prefix=filename_prefix, test_dataset=test_data)

		tf.reset_default_graph()

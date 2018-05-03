from __future__ import print_function
from __future__ import division 

from collections import Counter
import glob

import numpy as np
from scipy.spatial.distance import cdist
import skimage.io as imio
import skimage.transform as imtransform
import tensorflow as tf
import tensorflow.contrib.slim as slim
#import matplotlib.pyplot as plot

###### configuration ###########################################################

config = {
    "num_runs": 10,
    "batch_size": 10,
    "base_learning_rate": 0.001,
    "base_lr_decay": 0.9,
    "base_lr_decays_every": 4,
    "base_lr_min": 0.0001,
    "new_learning_rate": 0.0001,
    "new_lr_decay": 1.,
    "new_lr_decays_every": 1,
    "new_lr_min": 1e-6,
    "base_training_epochs": 1, # TODO
    "new_training_epochs": 100,
    "new_batch_num_replay": 8, # how many of batch of new items are replays
                                # if replay is on
    "SW_by": "reps", # one of "images" or "reps", what feature space to do
		       # the similarity weighting in
    "softmax_temp": 1, # temperature for SWIL replay softmax
    "SWIL_epsilon": 1e-5, # small constant in denominator for numerical
			  # stabiility when normalizing by sd
    "OMG_train_dir": "./omniglot_data/images_background/", # omniglot training directory
    "OMG_test_dir": "./omniglot_data/images_evaluation/", # omniglot testing directory
    "im_size": 50, # image width/height to resize image
    "output_path": "./omniglot_results_euclidean/",
    "nobias": True, # no biases
    "similarity_by": "euclidean", # Cosine distance or euclidean
    "layer_sizes": [256, 64, 32, 64, 256]
}

###### OMNIGLOT data loading and manipulation #####################################

def load_omniglot_data(base_dir):
    dataset = {"alphabets": [], "characters": [], "images": []}
    for i, alphabet_dir in enumerate(glob.glob(base_dir + '*')):
        for char_dir in glob.glob(alphabet_dir + '/*'):
            char_num = int(char_dir[-2:])
            for image_file in glob.glob(char_dir + '/*.png'):
                image = imio.imread(image_file)
                im_size = config["im_size"] 
                image = 1.-imtransform.resize(image, [im_size, im_size])
                image = image.flatten()
                dataset["alphabets"].append(i)
                dataset["characters"].append(char_num)
                dataset["images"].append(image)
    dataset["alphabets"] = np.array(dataset["alphabets"])
    dataset["characters"] = np.array(dataset["characters"])
    dataset["images"] = np.array(dataset["images"])
    return dataset


train_data = load_omniglot_data(config["OMG_train_dir"]) 
test_data = load_omniglot_data(config["OMG_test_dir"]) 

###### Build model func ########################################################

def softmax(x, T=1):
    """Compute the softmax function at temperature T"""
    if T != 1:
        x /= T
    x -= np.amax(x)
    x = np.exp(x)
    x /= np.sum(x)
    if not(np.any(x)): # handle underflow
        x = np.ones_like(x)/len(x) 
    return x

def to_unit_rows(x):
    """Converts row vectors of a matrix to unit vectors"""
    return x/np.expand_dims(np.sqrt(np.sum(x**2, axis=1)), -1)

def _display_image(x):
    x = np.reshape(x, [config["im_size"], config["im_size"]])
    plot.figure()
    plot.imshow(x, vmin=0, vmax=1)

class OMG_autoenc(object):
    """OMG autoencoder architecture, with or without replay buffer"""

    def __init__(self, replay_type, layer_sizes):
        """Create a OMG_autoenc model. 
            replay_type: one of ["None", "Random", "SWIL"]
            layer_sizes: list of the hidden layer sizes of the model
        """
        self.replay_type = replay_type
        self.base_lr = config["base_learning_rate"]
        self.new_lr = config["new_learning_rate"]

        flat_im_size = config["im_size"] ** 2

        self.input_ph = tf.placeholder(tf.float32, [None, flat_im_size])
        self.lr_ph = tf.placeholder(tf.float32)

        self.bottleneck_size = min(layer_sizes)

	# small weight initializer
	weight_init = tf.contrib.layers.variance_scaling_initializer(factor=0.3, mode='FAN_AVG')
	

        net = self.input_ph
	bottleneck_layer_i = len(layer_sizes)//2
        for i, h_size in enumerate(layer_sizes):
	    if config["nobias"]:
	      net = slim.layers.fully_connected(net, h_size, activation_fn=tf.nn.relu,
						weights_initializer=weight_init,
						biases_initializer=None)
	    else:
	      net = slim.layers.fully_connected(net, h_size, activation_fn=tf.nn.relu,
						weights_initializer=weight_init)
            if i == bottleneck_layer_i: 
                self.bottleneck_rep = net
	if config["nobias"]:
	    self.output = slim.layers.fully_connected(net, flat_im_size, activation_fn=tf.nn.sigmoid,
						      weights_initializer=weight_init,
						      biases_initializer=None)
	else:
	    self.output = slim.layers.fully_connected(net, flat_im_size, activation_fn=tf.nn.sigmoid,
						      weights_initializer=weight_init)
						  
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
                    fout.write(("epoch," + ', '.join(["%i"] * 30) + "\n") % tuple(range(30)))
		    losses = self.eval(test_dataset)
		    fout.write(("0, " + ', '.join(["%f"] * 30) +  "\n" ) % tuple(losses))
	    with open(config["output_path"] + log_file_prefix + "base_train_losses.csv", "w") as fout:
                fout.write(("epoch," + ', '.join(["%i"] * 30) + "\n") % tuple(range(30)))
		losses = self.eval(dataset)
                fout.write(("0, " + ', '.join(["%f"] * 30) +  "\n") % tuple(losses))

        batch_size = config["batch_size"]
        for epoch in range(1, nepochs + 1):
            order = np.random.permutation(len(dataset["alphabets"]))
            for batch_i in xrange(len(dataset["alphabets"])//batch_size):
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
			fout.write((("%i, " % epoch) + ', '.join(["%f"] * 30) + "\n") % tuple(losses))
		with open(config["output_path"] + log_file_prefix + "base_train_losses.csv", "a") as fout:
		    losses = self.eval(dataset)
                    fout.write((("%i, " % epoch) + ', '.join(["%f"] * 30) + "\n") % tuple(losses))

	    # update lr
	    if epoch > 0 and epoch % config["base_lr_decays_every"] == 0 and self.base_lr > config["base_lr_min"]: 
		self.base_lr *= config["base_lr_decay"]
         
    def new_data_train(self, new_dataset, old_dataset=None, nepochs=100, test_dataset=None, log_file_prefix=None):
        """Assuming the model has been trained on old_dataset, tune on
        new_dataset, possibly interleaving from old"""
        softmax_temp = config["softmax_temp"]
	SWIL_epsilon = config["SWIL_epsilon"]
        batch_size = config["batch_size"]
        if self.replay_type != "None":
            old_batch_size = config["new_batch_num_replay"]
            new_batch_size = batch_size - old_batch_size 
        
        if log_file_prefix is not None:
            if self.replay_type != "None": 
                with open(config["output_path"] + log_file_prefix + "replay_labels_encountered.csv", "w") as fout:
                    fout.write(("epoch," + ', '.join(["%i"] * 30) + "\n") % tuple(range(30)))
            if test_dataset is not None:
                with open(config["output_path"] + log_file_prefix + "new_test_losses.csv", "w") as fout:
                    fout.write(("epoch," + ', '.join(["%i"] * 30) + "\n") % tuple(range(30)))
		    losses = self.eval(test_dataset)
                    fout.write(("0, " + ', '.join(["%f"] * 30) +  "\n") % tuple(losses))
	    with open(config["output_path"] + log_file_prefix + "new_base_losses.csv", "w") as fout:
                fout.write(("epoch," + ', '.join(["%i"] * 30) + "\n") % tuple(range(30)))
		losses = self.eval(old_dataset)
                fout.write(("0, " + ', '.join(["%f"] * 30) +  "\n") % tuple(losses))
	    with open(config["output_path"] + log_file_prefix + "new_train_losses.csv", "w") as fout:
                fout.write(("epoch," + ', '.join(["%i"] * 30) + "\n") % tuple(range(30)))
		losses = self.eval(new_dataset)
                fout.write(("0, " + ', '.join(["%f"] * 30) +  "\n") % tuple(losses))

        for epoch in range(1, nepochs + 1):
            if self.replay_type != "None":
                replay_labels_encountered = Counter()

            order = np.random.permutation(len(new_dataset["alphabets"]))
            if self.replay_type == "Random":
                old_order = np.random.permutation(len(old_dataset["alphabets"]))
            elif self.replay_type == "SWIL":
		if config["SW_by"] == "reps":
		    old_dataset_reps = self.get_reps(old_dataset["images"])
		else:
		    old_dataset_reps = old_dataset["images"]
		# standardize
		old_dataset_reps_means = np.mean(old_dataset_reps, axis=0)
		old_dataset_reps_sds = np.std(old_dataset_reps, axis=0)
		old_dataset_reps = (old_dataset_reps - old_dataset_reps_means)/(old_dataset_reps_sds + SWIL_epsilon) 
		if config["similarity_by"] == "cosine":
		  # normalize
		  old_dataset_reps = to_unit_rows(old_dataset_reps)

            for batch_i in xrange(len(new_dataset["alphabets"])//batch_size):
                if self.replay_type == "None":
                    this_batch_images = new_dataset["images"][order[batch_i*batch_size:(batch_i+1)*batch_size], :]
                elif self.replay_type == "Random":
                    this_batch_images = np.concatenate(
                        [new_dataset["images"][order[batch_i*new_batch_size:(batch_i+1)*new_batch_size], :],
                         old_dataset["images"][old_order[batch_i*old_batch_size:(batch_i+1)*old_batch_size], :]])
                    replay_labels_encountered.update(old_dataset["alphabets"][old_order[batch_i*old_batch_size:(batch_i+1)*old_batch_size]])
                else: # SWIL
                    this_batch_new = new_dataset["images"][order[batch_i*new_batch_size:(batch_i+1)*new_batch_size], :]
		    if config["SW_by"] == "reps":
			this_batch_new_reps = self.get_reps(this_batch_new)
		    else:
			this_batch_new_reps = this_batch_new
                    # standardize
                    this_batch_new_reps_reps = (this_batch_new_reps - old_dataset_reps_means)/(old_dataset_reps_sds + SWIL_epsilon)
		    if config["similarity_by"] == "cosine":
			# normalize
			this_batch_new_reps = to_unit_rows(this_batch_new_reps)
			dots = np.dot(old_dataset_reps, this_batch_new_reps.transpose())
			probabilities = softmax(np.amax(dots, axis=1), T=softmax_temp) 
		    else: 
			dists = cdist(this_batch_new_reps, old_dataset_reps, metric="euclidean") 
			probabilities = softmax(-dists, T=softmax_temp) 
			  
                
                    # note that this is actually pretending that sampling with
                    # replacement is sampling without, but on a dataset this 
                    # large that shouldn't really be an issue unless the temp
                    # is too small...
                    replay_samples = np.where(np.random.multinomial(old_batch_size, probabilities)) 
                    
                    this_batch_images = np.concatenate(
                        [this_batch_new,
                         old_dataset["images"][replay_samples]])

                    replay_labels_encountered.update(old_dataset["alphabets"][replay_samples])

                self.sess.run(self.train, feed_dict={
                        self.input_ph: this_batch_images,
                        self.lr_ph: self.new_lr 
                    })

	    # eval
	    if log_file_prefix is not None:
		if self.replay_type != "None": 
		    with open(config["output_path"] + log_file_prefix + "replay_labels_encountered.csv", "a") as fout:
			counts = [replay_labels_encountered[i] for i in range(30)] 
                        fout.write((("%i, " % epoch) + ', '.join(["%i"] * 30) + "\n") % tuple(counts))
		if test_dataset is not None:
		    with open(config["output_path"] + log_file_prefix + "new_test_losses.csv", "a") as fout:
			losses = self.eval(test_dataset)
                        fout.write((("%i, " % epoch) + ', '.join(["%f"] * 30) + "\n") % tuple(losses))
		with open(config["output_path"] + log_file_prefix + "new_train_losses.csv", "a") as fout:
		    losses = self.eval(new_dataset)
                    fout.write((("%i, " % epoch) + ', '.join(["%f"] * 30) + "\n") % tuple(losses))
		with open(config["output_path"] + log_file_prefix + "new_base_losses.csv", "a") as fout:
		    losses = self.eval(old_dataset)
                    fout.write((("%i, " % epoch) + ', '.join(["%f"] * 30) + "\n") % tuple(losses))

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
        losses_summarized = [np.sum(losses[dataset["alphabets"] == i])/np.sum(dataset["alphabets"] == i) for i in range(30)]
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
    for left_out_alphabet in range(10): 
	for replay_type in ["SWIL", "Random",  "None"]:
	    for temperature in [1., 0.5, 0.1, 2., 10.]:
		if temperature != 1 and replay_type != "SWIL":
		    continue 
		config["softmax_temp"] = temperature # ugly
		filename_prefix = "OMG_run%i_lo%i_m_%s_" %(run,
						       left_out_alphabet,
						       replay_type)
		if replay_type == "SWIL":
		    filename_prefix += "swby_%s_t_%.3f_" % (config["SW_by"], config["softmax_temp"]) 
		print(filename_prefix)
		np.random.seed(run)
		tf.set_random_seed(run)


		model = OMG_autoenc(replay_type=replay_type,
				      layer_sizes=config["layer_sizes"])

		indices = test_data["alphabets"] == left_out_alphabet
		this_base_data = train_data 
		this_new_data = {"alphabets": test_data["alphabets"][indices],
                                 "characters": test_data["characters"][indices],
				 "images": test_data["images"][indices]}
		model.base_train(this_base_data, config["base_training_epochs"],
				 log_file_prefix=filename_prefix)
		model.new_data_train(this_new_data, this_base_data, config["new_training_epochs"],
				     log_file_prefix=filename_prefix)

		tf.reset_default_graph()

"""Builds the model

Implements the inference/loss/training pattern for model building.

1. inference() - Builds the model as far as is required for running the network
forward to make predictions.
2. loss() - Adds to the inference model the layers required to generate loss.
3. training() - Adds to the loss model the Ops required to generate and
apply gradients.

Note: this file is meant to be used with train.py (do not run this file).
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf

# CIFAR-10 has 10 classes
NUM_CLASSES = 10

# CIFAR-10 images are 32x32x3
IMAGE_SIZE = 32
IMAGE_CHANNELS = 3
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE * IMAGE_CHANNELS

def _create_kernel(name, shape):
	"""Creates a kernel variable

	Args:
	name: local name for variable
	shape: an array describing the shape

	Returns:
	Tensor
	"""
	initializer = tf.truncated_normal_initializer(stddev=5e-2)
	var = tf.get_variable(name, shape, initializer=initializer)

	return var

def inference(images):
	"""Build the model up to where it may be used for inference.

	Args:
	images: Images placeholder.

	Returns:
	softmax_linear: Output tensor with the computed logits.
	"""

	# CONV 1
	with tf.name_scope('conv1'):
		kernel = _create_kernel('weights', shape=[3, 3, 3, 4])
		# Stride of 1 in all dimensions (note the two middle elements are horiz/vert strides)
		stride = [1, 1, 1, 1]
		conv = tf.nn.conv2d(images, kernel, stride, padding='SAME')
		biases = tf.get_variable('biases', [4], tf.constant_initializer(0.0))
		pre_activation = tf.nn.bias_add(conv, biases)
		conv1 = tf.nn.relu(pre_activation, name='conv1')

	# POOL 1
	# Only the middle elements matter (the rest correspond to the batch etc)
	pool_shape = [1, 2, 2, 1]
	pool_stride = [1, 2, 2, 1]
	pool1 = tf.nn.max_pool(conv1, pool_shape, pool_stride, padding='SAME')

	# FC 1
	with tf.name_scope('fc1'):
		weights = _create_kernel('weights', [16, 16, 4, 10])
		stride = [1, 1, 1, 1]
		conv = tf.nn.conv2d(pool1, kernel, stride, padding='VALID')
		biases = tf.get_variable('biases', [10], tf.constant_initializer(0.0))
		pre-activation = tf.nn.bias_add(conv, biases)
		fc1 = tf.nn.relu(pre-activation, name='fc1')

	return fc1
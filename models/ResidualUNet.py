from __future__ import division
import os,time,cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def conv_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv block for Encoder-Decoder
	Apply successivly Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d(inputs, n_filters, kernel_size, activation_fn=None, normalizer_fn=None)
	out = tf.nn.relu(slim.batch_norm(conv, fused=True))
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def conv_transpose_block(inputs, n_filters, kernel_size=[3, 3], dropout_p=0.0):
	"""
	Basic conv transpose block for Encoder-Decoder upsampling
	Apply successivly Transposed Convolution, BatchNormalization, ReLU nonlinearity
	Dropout (if dropout_p > 0) on the inputs
	"""
	conv = slim.conv2d_transpose(inputs, n_filters, kernel_size=[3, 3], stride=[2, 2], activation_fn=None)
	out = tf.nn.relu(slim.batch_norm(conv))
	if dropout_p != 0.0:
	  out = slim.dropout(out, keep_prob=(1.0-dropout_p))
	return out

def build_residual_unet(inputs, num_classes, preset_model = "ResidualUNet", dropout_p=0.5, scope=None):
	"""
	Builds the Residual U-Net model. Inspired by SegNet with some modifications
	Includes skip connections

	Arguments:
	  inputs: the input tensor
	  n_classes: number of classes
	  dropout_p: dropout rate applied after each convolution (0. for not using)

	Returns:
	  ResidualUNet model
	"""
        
	#####################
	# Downsampling path #
	#####################
	
	net = conv_block(inputs, 64)
	res = net
	net = conv_block(net, 64)
	net = tf.add(net, res)
	skip_1 = net
        
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

	net = conv_block(net, 128)
	res = net
	net = conv_block(net, 128)
	net = tf.add(net, res)
	skip_2 = net
	
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

	net = conv_block(net, 256)
	res = net
	net = conv_block(net, 256)
	net = conv_block(net, 256)
	net = tf.add(net, res)
	skip_3 = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

	net = conv_block(net, 512)
	res = net
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = tf.add(net, res)
	skip_4 = net
	net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')

	net = conv_block(net, 512)
	res = net
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = tf.add(net, res)
	
	#net = slim.pool(net, [2, 2], stride=[2, 2], pooling_type='MAX')


	#####################
	# Upsampling path #
	#####################
	
	#net = conv_transpose_block(net, 512)

	net = conv_block(net, 512)
	res = net
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = tf.add(net, res)

	net = conv_transpose_block(net, 512)
	net = tf.concat([net, skip_4], 3 )

	net = conv_block(net, 512)
	res = net
	net = conv_block(net, 512)
	net = conv_block(net, 512)
	net = tf.add(net, res)

	net = conv_transpose_block(net, 256)
	net = tf.concat([net, skip_3], 3 )

	net = conv_block(net, 256)
	res = net
	net = conv_block(net, 256)
	net = conv_block(net, 256)
	net = tf.add(net, res)

	net = conv_transpose_block(net, 128)
	net = tf.concat([net, skip_2], 3 )

	net = conv_block(net, 128)
	res = net
	net = conv_block(net, 128)
	net = tf.add(net, res)

	net = conv_transpose_block(net, 64)
	net = tf.concat([net, skip_1], 3 )

	net = conv_block(net, 64)
	res = net
	net = conv_block(net, 64)
	net = tf.add(net, res)

	#####################
	#      Softmax      #
	#####################
	net = slim.conv2d(net, num_classes, [1, 1], activation_fn=slim.softmax, scope='logits')
	return net

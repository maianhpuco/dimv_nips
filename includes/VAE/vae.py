import argparse
import logging
import os
import pickle
import random
import shutil

from matplotlib import pyplot as plt

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds
import tensorflow_probability as tfp

tf.enable_v2_behavior()


class Encoder(tf.keras.layers.Layer):
  """VAE encoder."""

  def __init__(self, input_shape):
    """Creates an instance of Encoder.

    Returns:
      Encoder instance.
    """

    super(Encoder, self).__init__()

    if DATASET == 'MNIST':
      encoder_layers = [(32, 3, 2), (64, 3, 2)]
    else:
      encoder_layers = [(40, 3, 2), (60, 3, 2), (60, 5, 2)]

    self.conv_layers = []
    for i, (num_filters, kernel_size, strides) in enumerate(encoder_layers):
      if i == 0:
        self.conv_layers.append(tf.keras.layers.Conv2D(
            num_filters, kernel_size, strides=strides, activation=tf.nn.relu,
            data_format='channels_last', input_shape=input_shape,
            padding='SAME'))
      else:
        self.conv_layers.append(tf.keras.layers.Conv2D(
            num_filters, kernel_size, strides=strides, activation=tf.nn.relu,
            data_format='channels_last', padding='SAME'))

    self.mu_proj = tf.keras.layers.Dense(Z_DIM, activation=None)
    self.sigma_proj = tf.keras.layers.Dense(Z_DIM, activation=tf.math.softplus)

  def call(self, x):
    """Computes the forward pass through the Encoder.

    Args:
      x: `Tensor`. 4-D `Tensor` of shape [batch_size, height, width, depth]
        containing the input images.

    Returns:
      A tuple of `Tensors` of shape [batch_size, z_dim] the mean and sigma
      parameters of a Gaussian distribution.
    """
    for layer in self.conv_layers:
      x = layer(x)

    x = tf.keras.layers.Flatten()(x)

    return (self.mu_proj(x), (self.sigma_proj(x) + 1e-3)) 
class Decoder(tf.keras.layers.Layer):
  """VAE decoder."""

  def __init__(self):
    """Creates an instance of Decoder.

    Returns:
      Decoder instance.
    """
    super(Decoder, self).__init__()

    if DATASET == 'MNIST':
      self.dense = tf.keras.layers.Dense(7*7*20, activation=tf.nn.relu)
      self.reshape_shape = [-1, 7, 7, 20]
      decoder_layers = [(40, 5, 2), (20, 5, 2)]
      fine_tune_layers = [(10, 5, 1), (10, 5, 1)]
      assert LIKELIHOOD == 'BERNOULLI'
      last_layer = (1, 3, 1)
    else:
      self.dense = tf.keras.layers.Dense(4*4*60, activation=tf.nn.relu)
      self.reshape_shape = [-1, 4, 4, 60]
      decoder_layers = [(60, 3, 2), (60, 3, 2), (40, 5, 2)]
      fine_tune_layers = [(30, 5, 1), (30, 5, 1)]
      if LIKELIHOOD == 'LOGISTIC_MIXTURE':
        last_layer = (9 * LOGISTIC_MIXTURE_COMPONENTS, 3, 1)
      else:
        last_layer = (3, 3, 1)

    self.decoder_layers = []
    for i, (num_filters, kernel_size, strides) in enumerate(decoder_layers):
      self.decoder_layers.append(tf.keras.layers.Conv2DTranspose(
          num_filters, kernel_size, strides=strides, activation=tf.nn.relu,
          data_format='channels_last', padding='SAME'))
    
    self.fine_tune_layers = []
    for i, (num_filters, kernel_size, strides) in enumerate(fine_tune_layers):
      self.fine_tune_layers.append(tf.keras.layers.Conv2D(
          num_filters, kernel_size, strides=strides, activation=tf.nn.relu,
          data_format='channels_last', padding='SAME'))
    
    self.last_layer = tf.keras.layers.Conv2D(
          last_layer[0], last_layer[1], strides=last_layer[2],
          activation=None, data_format='channels_last', padding='SAME')

def call(self, x, b=None):
    """Computes the forward pass through the Decoder.

    Args:
      x: `Tensor`. 4-D `Tensor` of shape [batch_size, height, width, depth]
        containing the input images.

    Returns:
      Tuple of three `Tensors` (mean_logit, scale_logit, pi_logit)
    """
    x = self.dense(x)
    x = tf.reshape(x, self.reshape_shape)
    for layer in self.decoder_layers:
      x = layer(x)

    if b is not None:
      x = tf.concat([x, b], axis=-1)
    
    for layer in self.fine_tune_layers:
      x = layer(x)
    
    x = self.last_layer(x)

    if LIKELIHOOD == 'LOGISTIC_MIXTURE':
      mean_logit = []
      scale_logit = []
      pi_logit = []
      img_channels = 3
      k = LOGISTIC_MIXTURE_COMPONENTS
      for i in range(img_channels):
        mean_logit.append(x[:, :, :, i*k:(i+1)*k])
        scale_logit.append(x[:, :, :,
                             (img_channels*k + i*k):(img_channels*k + (i+1)*k)])
        pi_logit.append(x[:, :, :,
                          (2*img_channels*k + i*k):(2*img_channels*k + (i+1)*k)])
    else:
        mean_logit = x
        scale_logit = None
        pi_logit = None
    
    return mean_logit, scale_logit, pi_logit

class VAE(tf.keras.Model):
  def __init__(self, input_shape):
    super(VAE, self).__init__()
    self.encoder = Encoder(input_shape)
    self.decoder = Decoder()

  def call(self, inputs, decoder_b=None):
    mu, sigma = self.encoder(inputs)
    q_z = tfp.distributions.Normal(mu, sigma)
    
    z_sample = q_z.sample()
    return self.decoder(z_sample, b=decoder_b), q_z, z_sample 




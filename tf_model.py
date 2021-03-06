import os
import sys
from functools import wraps
import numpy as np
import tensorflow as tf
import pandas as pd

from .base_model import BaseModel

class TFModel(BaseModel):
    """ Base class for all tensorflow models. """

    activations = {'relu': tf.nn.relu,
                   'sigmoid': tf.nn.sigmoid,
                   'tanh': tf.nn.tanh,
                   'softmax': tf.nn.softmax,
                   'linear': tf.identity}

    def __new__(cls, name, *args, **kwargs):
        """ Add necessary attributes to object of tensorflow model. """
        instance = super(TFModel, cls).__new__(cls)
        instance.__scope_name = name
        instance.__graph = tf.Graph()
        instance.__phase = tf.placeholder(tf.bool)
        return instance

    @property
    def graph(self):
        """ Return graph object of the current model. """
        return self.__graph

    @property
    def phase(self):
        """ Return tensorflow placeholder representing learning phase.

        Returns tf.placeholder(tf.bool).
        """
        return self.__phase

    @property
    def scope_name(self):
        """ Return name of model's main scope.

        Returns string.
        """
        return self.__scope_name

    @staticmethod
    def num_channels(input_tensor):
        """ Return channels dimension of the input tensor.

        Args:
        - input_tensor: tf.Variable, input tensor.

        Return:
        - number of channels, int;
        """
        return input_tensor.get_shape().as_list()[-1]

    @staticmethod
    def batch_size(input_tensor):
        """ Return batch size of the input tensor represented by first dimension.

        Args:
        - input_tensor: tf.Variable, input_tensor.

        Return:
        - number of items in the batch, int;
        """
        return input_tensor.get_shape().as_list()[0]

    @staticmethod
    def get_shape(input_tensor):
        """ Return full shape of the input tensor represented by tuple of ints.

        Args:
        - input_tensor: tf.Variable, input_tensor.

        Return:
        - shape of input_tensor, tuple(int);
        """
        return input_tensor.get_shape().as_list()

    def maxpool_2d(self, input_tensor, kernel_size, name, strides=(1, 1), padding='SAME'):
        """ Wrap input tensor with maxpooling 2d layer and return result.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - kernel_size: tuple(int, int) representing kernel size of pooling;
        - name: name of this layer's scope;
        - strides: tuple(int, int) representing strides along x and y axes;
        - padding: padding mode for pooling operation;

        Returns:
        - output tensor, tf.Variable;
        """
        with tf.variable_scope("MaxPool2D", name):
            out_layer = tf.nn.max_pool(input_tensor, ksize=(1, *kernel_size, 1),
                                       strides=(1, *strides, 1), padding=padding)
        return out_layer

    def conv2d(self, input_tensor, filters, kernel_size, name,
               activation='linear', strides=(1, 1), padding='SAME'):
        """ Wrap input tensor with 2D convolutional layer and return result.

        Args:
        - filters: int, number of filters in the output tensor;
        - input_tensor: tf.Variable, input tensor;
        - kernel_size: tuple(int, int) representing kernel size of convolution;
        - name: name of this layer's scope;
        - activation: str, activation to put after convolution;
        - strides: tuple(int, int) representing strides along x and y axes;
        - padding: padding mode for pooling operation;

        Returns:
        - output tensor, tf.Variable;
        """
        with tf.variable_scope("Conv2D", name):
            init_w = tf.truncated_normal(shape=(*kernel_size,
                                                self.num_channels(input_tensor),
                                                filters),
                                         dtype=tf.float32)

            w = tf.Variable(init_w, name='W')
            b = tf.Variable(tf.random_uniform(shape=(filters, ), dtype=tf.float32))

            out_layer = tf.nn.conv2d(input_tensor, w, strides=(1, *strides, 1),
                                     padding=padding,
                                     name='conv_2d_op')
            out_layer = self.activations[activation](out_layer)
        return out_layer

    def dense(self, input_tensor, units, name, activation='linear'):
        """ Wrap input tensor with dense layer.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - units: int, number of units in the output tensor;
        - name: str, name of the dense layer's scope;
        - activation: str, activation to put after convolution;

        Return:
        - output tensor, tf.Variable;
        """
        with tf.variable_scope('Dense', name):
            init_w = tf.truncated_normal(shape=(self.num_channels(input_tensor),
                                                units), dtype=tf.float32)
            w = tf.Variable(init_w)
            b = tf.Variable(tf.random_uniform(shape=(units, ), dtype=tf.float32))

            out_layer = tf.matmul(input_tensor, w) + b
            out_layer = self.activations[activation](out_layer)
        return out_layer

    def identity(self, input_tensor, name):
        """ Create an alias with given name for input_tensor.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - name: str, name of alias;

        Return:
        - alias of input tensor, tf.Variable;
        """
        return tf.identity(input_tensor, name=name)

    @wraps(tf.layers.flatten)
    def flatten(self, input_tensor, **kwargs):
        return tf.contrib.flatten(input_tensor, **kwargs)

    @wraps(tf.layers.batch_normalization)
    def batch_norm(self, *args, **kwargs):
        return tf.layers.batch_normalization(*args, **kwargs, training=self.phase)

    @wraps(tf.layers.dropout)
    def dropout(self, *args, **kwargs):
        return tf.layers.dropout(*args, **kwargs, training=self.phase)

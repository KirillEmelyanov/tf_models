import os
import sys
from functools import wraps
import numpy as np
import tensorflow as tf
import pandas as pd

class TFModel(object):
    """ Base class for all tensorflow models. """

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
        """ Return wraps input tensor with maxpooling 2d layer and returns result.

        Args:
        - input_tensor: tf.Variable, input tensor;
        - kernel_size: tuple(int, int) representing kernel size of pooling;
        - name: name of this layer scope;
        - strides: tuple(int, int) representing strides along x and y axes;
        - padding: padding mode for pooling operation;

        Returns:
        - output tensor, tf.Variable;
        """
        with tf.variable_scope("MaxPool2D", name):
            out_layer = tf.nn.max_pool(input_tensor, ksize=(1, *kernel_size, 1),
                                       strides=(1, *strides, 1), padding=padding)
        return out_layer

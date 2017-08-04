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

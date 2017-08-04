from functools import wraps
import tensorflow as tf


def model(method):
    """ Wrap method of TFModel to apply all ops in context of model's graph and scope. """
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self.graph.as_default():
            with tf.variable_scope(self.scope):
                result = method(self, *args, **kwargs)
        return result
    return wrapped

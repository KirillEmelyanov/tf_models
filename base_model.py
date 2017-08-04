class BaseModel(object):

    def save(self, *args, **kwargs):
        """ Save model. """
        raise NotImplementedError

    def load(self, *args, **kwargs):
        """ Load model from hard drive. """
        raise NotImplementedError

    def fit(self, *args, **kwargs):
        """ Fit model on input data. """
        raise NotImplementedError

    def predict(self, *args, **kwargs):
        """ Get predictions of model on input data. """
        raise NotImplementedError

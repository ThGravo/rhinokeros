import numpy as np


class Model(object):
    def __init__(self, input_space, output_space):
        self.input_space = input_space
        self.output_space = output_space

    def predict(self, x):
        return 0

    def train(self, x, y):
        return


class RegressionModel(Model):
    """Simplest regression model possible returning the moving average"""

    def __init__(self, input_space, output_space):
        super().__init__(input_space, output_space)
        self.theta = 0
        self.num_samples_seen = 0

    def predict(self, x):
        return self.theta

    def train(self, x, y):
        self.theta = (y + self.num_samples_seen * self.theta) / (self.num_samples_seen + 1)
        self.num_samples_seen += 1


class ClassificationModel(Model):
    """Simplest classification model possible predicting the most frequent class"""

    def __init__(self, input_space, output_space):
        super().__init__(input_space, output_space)
        self.hist = np.zeros(output_space.n)

    def predict(self, x):
        return np.argmax(self.hist)

    def train(self, x, y):
        self.hist[max(0, np.round(y))] += 1

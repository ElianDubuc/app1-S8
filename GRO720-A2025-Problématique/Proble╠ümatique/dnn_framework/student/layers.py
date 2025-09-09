import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        #sqrt((2 / (input + output)))   Permet d'avoir une uniformité entre les variances des données e/s, init les params à des valeurs plus cohérentes que du hasard pur
        #Random simple fonctionne aussi mais apporte une plus faible précision au final
        variance_w = 2 / (input_count + output_count)
        self.weights = np.random.randn(output_count, input_count) * np.sqrt(variance_w)

        variance_b = 2 / output_count
        self.biases = np.random.randn(1, output_count) * np.sqrt(variance_b)


    def get_parameters(self):
        return {'w' : self.weights,  'b': self.biases}
        raise NotImplementedError()

    def get_buffers(self):
        return {}
        raise NotImplementedError()

    def forward(self, x):
        a=(x @ self.weights.T) + self.biases #y = x*Wt + b
        return a, x
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        dW = output_grad.T @ cache  # gradient wrt W
        db = np.sum(output_grad, axis=0)  # gradient wrt b
        dX = output_grad @ self.weights  # gradient wrt X

        return dX, {'w': dW, 'b':db}
        raise NotImplementedError()


class BatchNormalization(Layer):
    """
    This class implements a batch normalization layer.
    """

    def __init__(self, input_count, alpha=0.1):
        super().__init__()

        self.gamma = np.ones((input_count,))
        self.beta = np.zeros((input_count,))

        self.running_mean = np.zeros((input_count,))
        self.running_variance = np.ones((input_count,))

        self.alpha = alpha

        self.safety = 1e-07

    def get_parameters(self):
        return {'gamma': self.gamma, 'beta': self.beta}
        raise NotImplementedError()

    def get_buffers(self):
        return {'global_mean': self.running_mean, 'global_variance': self.running_variance}
        raise NotImplementedError()

    def forward(self, x):
        if self.is_training():
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)
        raise NotImplementedError()

    def _forward_training(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_variance = np.var(x, axis=0)

        x_normalized = (x - batch_mean) / (np.sqrt(batch_variance) + self.safety)
        y = self.gamma * x_normalized + self.beta

        self.running_mean = self.alpha * self.running_mean + (1 - self.alpha) * batch_mean
        self.running_variance = self.alpha * self.running_variance + (1 - self.alpha) * batch_variance

        cache = (x, x_normalized, batch_mean, batch_variance)
        return y, cache

        raise NotImplementedError()

    def _forward_evaluation(self, x):
        x_normalized = (x - self.running_mean) / (np.sqrt(self.running_variance) + self.safety)

        y = self.gamma * x_normalized + self.beta

        return y, None

    def backward(self, output_grad, cache):
        x, x_normalized, batch_mean, batch_variance = cache

        N = x.shape[0]

        dgamma = np.sum(output_grad * x_normalized, axis=0)
        dbeta = np.sum(output_grad, axis=0)

        dx_normalized = output_grad * self.gamma

        dvariance = np.sum(dx_normalized * (x - batch_mean) * -0.5 * np.power(batch_variance, -1.5), axis=0)

        dmean = np.sum(dx_normalized * -1.0 / np.sqrt(batch_variance), axis=0) + dvariance * np.mean(
            -2.0 * (x - batch_mean), axis=0)

        dx = (dx_normalized / np.sqrt(batch_variance)) + (dvariance * 2.0 * (x - batch_mean) / N) + (dmean / N)

        return dx, {'gamma': dgamma, 'beta': dbeta}

        raise NotImplementedError()


class Sigmoid(Layer):
    """
    This class implements a sigmoid activation function.
    """

    def get_parameters(self):
        params = {}
        return params
        raise NotImplementedError()

    def get_buffers(self):
        buffers = {}
        return buffers
        raise NotImplementedError()

    def forward(self, x):
        dict = {'x': x}
        a=1 / (1 + np.exp(-x))
        return a, dict
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        sigmoid = 1 / (1 + np.exp(-cache['x']))
        return output_grad * sigmoid * (1 - sigmoid), cache
        raise NotImplementedError()


class ReLU(Layer):
    """
    This class implements a ReLU activation function.

    < 0 -> 0

    """

    def get_parameters(self):
        params = {}
        return params
        raise NotImplementedError()

    def get_buffers(self):
        buffers = {}
        return buffers
        raise NotImplementedError()

    def forward(self, x):
        #dict en vue de rétropropag
        dict = {'x': x}
        a = np.maximum(0, x)
        return a, dict
        raise NotImplementedError()

    def backward(self, output_grad, cache):
        a = output_grad * (cache['x'] > 0)
        return a, cache
        raise NotImplementedError()

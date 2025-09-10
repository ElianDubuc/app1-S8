import numpy as np

from dnn_framework.layer import Layer


class FullyConnectedLayer(Layer):
    """
    This class implements a fully connected layer.
    """

    def __init__(self, input_count, output_count):
        #sqrt((2 / (input + output)))   Permet d'avoir une uniformité entre les variances des données e/s, init les params à des valeurs plus cohérentes que du hasard pur
        #Random simple fonctionne aussi mais apporte une plus faible précision au final
        self.weights = np.random.randn(output_count, input_count) * (2 / (input_count + output_count)) ** 0.5
        self.biases = np.random.randn(1, output_count) * (2 / output_count) ** 0.5

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

        self.alpha = alpha
        self.gamma = np.ones((input_count,))
        self.beta = np.zeros((input_count,))

        self.mbl_mean = np.zeros((input_count,))
        self.mbl_vari = np.ones((input_count,))

    def get_parameters(self):
        return {'gamma': self.gamma, 'beta': self.beta}
        raise NotImplementedError()

    def get_buffers(self):
        return {'global_mean': self.mbl_mean, 'global_variance': self.mbl_vari}
        raise NotImplementedError()

    def forward(self, x):
        if self.is_training():
            return self._forward_training(x)
        else:
            return self._forward_evaluation(x)
        raise NotImplementedError()

    def _forward_training(self, x):
        epch_mean = np.mean(x, axis=0)
        epch_variance = np.var(x, axis=0)

        xNorm = (x - epch_mean) / (np.sqrt(epch_variance) + 1e-07)
        y = self.gamma * xNorm + self.beta

        self.mbl_mean = self.alpha * self.mbl_mean + (1 - self.alpha) * epch_mean
        self.mbl_vari = self.alpha * self.mbl_vari + (1 - self.alpha) * epch_variance

        cache = (x, xNorm, epch_mean, epch_variance)
        return y, cache

        raise NotImplementedError()

    def _forward_evaluation(self, x):
        xNorm = (x - self.mbl_mean) / (np.sqrt(self.mbl_vari) + 1e-07)
        y = self.gamma * xNorm + self.beta

        return y, None

    def backward(self, output_grad, cache):
        x, xNorm, epch_mean, epch_variance = cache

        dgamma = np.sum(output_grad * xNorm, axis=0)
        dbeta = np.sum(output_grad, axis=0)

        dxNorm = output_grad * self.gamma
        dvariance = np.sum(dxNorm * (x - epch_mean) * -0.5 * np.power(epch_variance, -1.5), axis=0)
        dmean = np.sum(dxNorm * -1.0 / np.sqrt(epch_variance), axis=0) + dvariance * np.mean(-2.0 * (x - epch_mean), axis=0)

        N = x.shape[0]
        dx = (dxNorm / np.sqrt(epch_variance)) + (dvariance * 2.0 * (x - epch_mean) / N) + (dmean / N)

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

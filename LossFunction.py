import numpy as np

class LossFunction:
    def __init__(self, regularization_type=None, regularization_lambda=0.01):
        self.regularization_type = regularization_type
        self.regularization_lambda = regularization_lambda

    def mse(self, predictions, targets):
        return np.mean((predictions - targets) ** 2)
    
    def mse_derivative(self, predictions, targets):
        return 2 * (predictions - targets) / len(targets)
    
    def add_regularization(self, weights):
        if self.regularization_type == 'L2':
            return self.regularization_lambda * np.sum(np.square(weights))
        elif self.regularization_type == 'L1':
            return self.regularization_lambda * np.sum(np.abs(weights))
        return 0
    
    def apply_regularization(self, weights):
        if self.regularization_type == 'L2':
            return self.regularization_lambda * weights
        elif self.regularization_type == 'L1':
            return self.regularization_lambda * np.sign(weights)
        return 0

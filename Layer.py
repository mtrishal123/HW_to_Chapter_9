import numpy as np

class Layer:
    def __init__(self, input_size, output_size, dropout_rate=0.0):
        self.weights = np.random.randn(input_size, output_size)
        self.bias = np.random.randn(output_size)
        self.dropout_rate = dropout_rate
        self.dropout_mask = None

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.bias
        
        # Dropout during training
        if self.dropout_rate > 0:
            self.dropout_mask = (np.random.rand(*self.z.shape) > self.dropout_rate).astype(np.float32)
            self.z *= self.dropout_mask  # Apply dropout mask
            
        return self.z

    def backward(self, dL_dz, learning_rate):
        dL_dw = np.dot(self.inputs.T, dL_dz)
        dL_db = np.sum(dL_dz, axis=0)

        self.weights -= learning_rate * dL_dw
        self.bias -= learning_rate * dL_db

from Layer import Layer


class Model:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        self.layer1 = Layer(input_size, hidden_size, dropout_rate)
        self.layer2 = Layer(hidden_size, output_size, dropout_rate=0.0)  # No dropout in the output layer
    
    def forward(self, X, training=True):
        hidden_output = self.layer1.forward(X)
        
        # Only apply dropout during training
        if training and self.layer1.dropout_rate > 0:
            hidden_output *= self.layer1.dropout_mask
        
        output = self.layer2.forward(hidden_output)
        return output

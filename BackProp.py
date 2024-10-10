class BackProp:
    def __init__(self, model, loss_function):
        self.model = model
        self.loss_function = loss_function

    def backward(self, X_train, y_train, predictions, learning_rate):
        output_error = self.loss_function.mse_derivative(predictions, y_train)
        
        # Backpropagation through output layer
        hidden_output_error = self.model.layer2.backward(output_error, learning_rate)
        
        # Account for dropout during backpropagation through hidden layer
        if self.model.layer1.dropout_rate > 0:
            hidden_output_error *= self.model.layer1.dropout_mask
            
        # Backpropagation through hidden layer
        self.model.layer1.backward(hidden_output_error, learning_rate)

class GradDescent:
    def __init__(self, model, loss_function, learning_rate=0.01):
        self.model = model
        self.loss_function = loss_function
        self.learning_rate = learning_rate

    def train(self, X_train, y_train, epochs):
        for epoch in range(epochs):
            predictions = self.model.forward(X_train)
            loss = self.loss_function.mse(predictions, y_train)
            
            # Add regularization term to the loss
            for layer in self.model.layers:
                loss += self.loss_function.add_regularization(layer.weights)
            
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss:.4f}")
            
            # Backpropagation step
            self.backprop.backward(X_train, y_train, predictions, self.learning_rate)
            
            # Apply regularization on weights during gradient update
            for layer in self.model.layers:
                layer.weights -= self.loss_function.apply_regularization(layer.weights)

import numpy as np
'''
Phase 1 — Basic Mathematics
Implement:
NumPy-based matrix operations
Linear layers
Weights
Biases
ReLU
Forward propagation
Goal: Make a neural network capable of producing predictions.
'''

class Layer:
    def __init__(self):
        self.input = None  
        self.output = None

    def forward(self, input):  
        pass

    def backward(self, output_gradient, learning_rate):
        pass


class Linear(Layer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.zeros((output_size, 1))

    def forward(self, input):
        self.input = input
        output = np.dot(self.weights, self.input) + self.biases
        return output

    def backward(self, output_gradient, learning_rate):
        input_gradient = np.dot(self.weights.T, output_gradient)

        self.weights -= learning_rate * (output_gradient @ self.input.T)
        self.biases -= learning_rate * np.sum(output_gradient, axis=1, keepdims=True)

        return input_gradient

    def __str__(self):
        return f'Linear({self.output_size})'

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        super().__init__()
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

    
class ReLu(Activation):
    def __init__(self):
        
        relu = lambda x: np.maximum(0, x)
        relu_prime = lambda x: np.where(x > 0, 1, 0)

        super().__init__(relu, relu_prime)

class Tanh(Activation):
    def __init__(self):
        
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2

        super().__init__(tanh, tanh_prime)
    

def MSE(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def MSE_prime(y_true, y_pred):
    return -2 * (y_true - y_pred) / np.size(y_true)


class Model:
    def __init__(self, layers, lr=0.001):
        self.layers = layers
        self.lr = lr

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, output_grad, lr):
        for layer in reversed(self.layers):
            output_grad = layer.backward(output_grad, lr)

    def train(self, X, y, epochs=5):
        for epoch in range(epochs):
            for trainX, trainy in zip(X, y):
                trainX = trainX.reshape(self.layers[0].input_size, 1)

                output = self.forward(trainX)

                error = MSE(trainy, output)
                grad = MSE_prime(trainy, output)
                self.backward(grad, self.lr)

    def predict(self, input):
        return self.forward(np.array(input).reshape(self.layers[0].input_size, 1))

inputs = np.random.uniform(-1, 1, 10000).reshape(10000, 1)
f = lambda x: 2 ** x
labels = f(inputs)

layers = [Linear(1, 8), Tanh(), Linear(8, 4), Tanh(), Linear(4, 1)]
model = Model(layers)

model.train(inputs, labels)

print(model.predict(-0.8))
print(model.predict(-0.2))
print(model.predict(0.4))
print(model.predict(0.8))

    


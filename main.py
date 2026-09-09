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

class Linear(Layer):
    def __init__(self, i, o):
        super().__init__()
        self.weights = np.random.rand(i, o)
        self.biases = np.random.rand(1, o)

    def forward(self, input):
        output = input.dot(self.weights) + self.biases
        return output

class ReLu:
    def forward(self, input):
        relu = lambda x: 0 if x < 0 else x

        vectorize_relu = np.vectorize(relu)

        output = vectorize_relu(input)
        return output

class Model:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, input):
        x = input
        for layer in self.layers:
            x = layer.forward(x)

        return x

layers = [Linear(1, 2), ReLu()]

model = Model(layers)

print(model.forward(np.array(-3)))
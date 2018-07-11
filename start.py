#! /usr/bin python3
import numpy as np
import math

class Net(object):
    def __init__(self):
        self.inputSize = 3
        self.outputSize = 3
        self.hiddenSize = 5

        # Weights
        self.W1 = np.random.rand(self.inputSize, self.hiddenSize)
        self.W2 = np.random.rand(self.hiddenSize, self.outputSize)

        # Synapses
        self.z2 = np.zeros(self.inputSize, self.hiddenSize)
        self.z3 = np.zeros(self.hiddenSize, self.outputSize)

        # Activations

        self.a2 = np.zeros(self.inputSize, self.hiddenSize)

    def forward(self, X):
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        y_hat = self.sigmoid(self.z3, True)
        return y_hat

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to W and W2 for a given X and y:
        y_hat = self.forward(X)

        delta3 = np.multiply(-(y - y_hat), self.sigmoid(self.z3, True))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoid(self.z2, True)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    def sigmoid(self, x, derivative=False):
        return x * (1 - x) if derivative else 1 / (1 + np.exp(-x))

    def derivnonlin(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def costfunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        delta3 = np.multiply(-(y - self.yHat), self.derivnonlin(self.z3))
        djdW2 = np.dot(self.a2.T, delta3)
        delta2 = np.dot(delta3, self.W2.T) * self.derivnonlin(self.z2)
        djdW1 = np.dot(X.T, delta2)

        return djdW1, djdW2

trainingSet = [
    [255, 0, 0],
    [0, 255, 0],
    [0, 0, 255]
]


def train():
    pass


if __name__ == "__main__":
    train()

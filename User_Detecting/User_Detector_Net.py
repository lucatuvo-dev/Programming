#Analyzing handwritten numbers
#Each input has 784 pixels, layer 1 has 10 nodes, layer 2 has 10 nodes

import numpy as np
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test,y_test) = mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2]).T
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2]).T
x_train = x_train / 255
x_test = x_test / 255

def ReLU(Z):
    return np.maximum(0,Z)

def ReLU_deriv(Z):
    return (Z > 0).astype(float)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

class Neural_Net():
    def __init__(self):
        self.W1 = np.random.rand(10,784) * 0.01
        self.b1 = np.random.rand(10,1)
        self.W2 = np.random.rand(10,10) * 0.01
        self.b2 = np.random.rand(10,1)

    def forward_prop(self, X):
        Z1 = self.W1.dot(X) + self.b1
        A1 = ReLU(Z1)
        Z2 = self.W2.dot(A1) + self.b2
        A2 = softmax(Z2)
        return Z1, A1, Z2, A2
    
    def backward_prop(self, Z1, A1, Z2, A2, X, Y):
        m = X.shape[1]
        one_hot_Y = one_hot(Y)
        dZ2 = A2 - one_hot_Y
        dW2 = (1 / m) * dZ2.dot(A1.T)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
        dZ1 = self.W2.T.dot(dZ2) * ReLU_deriv(Z1)
        dW1 = (1 / m) * dZ1.dot(X.T)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
        return dW1, db1, dW2, db2
    
    def update_params(self, dW1, db1, dW2, db2, alpha):
        self.W1 = self.W1 - alpha * dW1
        self.b1 = self.b1 - alpha * db1
        self.W2 = self.W2 - alpha * dW2
        self.b2 = self.b2 - alpha * db2
    
    def get_predictions(self, A2):
        return np.argmax(A2, 0)
    
    def get_accuracy(self, predictions, Y):
        print(predictions, Y)
        return np.sum(predictions == Y) / Y.size
    
    def gradient_descent(self, X, Y, alpha, iterations):
        for i in range(iterations):
            Z1, A1, Z2, A2 = self.forward_prop(X)
            dW1, db1, dW2, db2 = self.backward_prop(Z1, A1, Z2, A2, X, Y)
            self.update_params(dW1, db1, dW2, db2, alpha)
            if i % 10 == 0:
                predictions = self.get_predictions(A2)
                accuracy = self.get_accuracy(predictions, Y)
                print("Iteration: ", i, "Accuracy: ", accuracy)


nn = Neural_Net()
nn.gradient_descent(x_train[:, :1000], y_train[:1000], 0.1, 100)










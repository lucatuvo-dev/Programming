import numpy as np

x_train = []
y_train = []
x_test = []
y_test = []
#x is the image, y is the corresponding label
#each x is a list in itself containing the pixel values

kernel = np.array([1,2,1], [2,4,2], [1,2,1])
print(kernel)


#Figure out the image scaling process using kernals



#Neural Net Stuff
def ReLU(x):
    return np.maximum(0,x)

def ReLU_derivative(x):
    return (x > 0).astype(float)

def softmax(Z):
    exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
    return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

#class Neural_Net():
    #def __init__(self):
        #Size of weights and biases will depend on size of processed image, number of layers, and number of nodes in input layers
        #self.W1 = np.random.rand()











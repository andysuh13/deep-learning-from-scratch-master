import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax, cross_entropy_error
from common.gradient import numerical_gradient

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

# train_size = x_train.shape[0] # 60000
# batch_size = 10
# batch_mask = np.random.choice(train_size, batch_size)
# x_batch = x_train[batch_mask]
# t_batch = t_train[batch_mask]

# lst = np.arange(5)
# print(lst) # [0 1 2 3 4]

# def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
#     x = init_x
#     for i in range(step_num):
#         grad = numerical_gradient(f, x)
#         x -= lr * grad
#     return x

# class simpleNet:
#     def __init__(self):
#         self.W = np.random.randn(2, 3)

#     def predict(self, x):
#         return np.dot(x, self.W)

#     def loss(self, x, t):
#         z = self.predict(x)
#         y = softmax(z)
#         loss = cross_entropy_error(y, t)
#         return loss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2']= weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.parmas['b2']
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)
        return y
    
    def loss(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss)
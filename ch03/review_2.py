import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
import pickle
from common.functions import sigmoid, softmax
import time

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

# 기본 구현
x, t = get_data()
network = init_network()
accuracy_cnt = 0

start_time = time.time()
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
end_time = time.time()

print("Time : " + str(end_time - start_time)) # 0.494...
print("Accuracy : " + str(float(accuracy_cnt) / len(x))) # 0.9352

# # 배치 처리 구현 (기본 구현에 비해 훨씬 빠르다.)
# x, t = get_data()
# network = init_network()
# batch_size = 100
# accuracy_cnt = 0

# start_time = time.time()
# for i in range(0, len(x), batch_size):
#     x_batch = x[i:i+batch_size]
#     y_batch = predict(network, x_batch)
#     p = np.argmax(y_batch, axis = 1)
#     accuracy_cnt += np.sum(p == t[i:i+batch_size])
# end_time = time.time()

# print("Time : " + str(end_time - start_time)) # 0.056...
# print("Accuracy : " + str(float(accuracy_cnt) / len(x))) # 0.9352

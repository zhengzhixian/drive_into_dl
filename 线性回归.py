import time
import torch

# define a timer class to record time
class Timer(object):
    """Record multiple running times."""
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        # start the timer
        self.start_time = time.time()

    def stop(self):
        # stop the timer and record time into a list
        self.times.append(time.time() - self.start_time)
        return self.times[-1]

    def avg(self):
        # calculate the average and return
        return sum(self.times)/len(self.times)

    def sum(self):
        # return the sum of recorded time
        return sum(self.times)

'''
# init variable a, b as 1000 dimension vector
n = 1000
a = torch.ones(n)
b = torch.ones(n)

#循环标量做加法
timer = Timer()
c = torch.zeros(n)
for i in range(n):
    c[i] = a[i] + b[i]
print('%.5f sec' % timer.stop())

#直接加法
timer.start()
d = a + b
'%.5f sec' % timer.stop()
'''


import torch
from IPython import display
from matplotlib import pyplot as plt
import numpy as np
import random

print(torch.__version__)

#####################线性回归 从零开始实现#################################

##生成数据集：生成一个1000个样本的数据集
# set input feature number
num_inputs = 2
# set example number
num_examples = 1000
# set true weight and bias in order to generate corresponded label
true_w = [2, -3.4]
true_b = 4.2
features = torch.randn(num_examples, num_inputs,dtype=torch.float32)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
labels += torch.tensor(np.random.normal(0, 0.01, size=labels.size()),dtype=torch.float32)
plt.show()

##读取数据集 yield 返回迭代器
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    # random read 10 samples
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        # the last time may be not enough for a whole batch
        j = torch.LongTensor(indices[i: min(i + batch_size, num_examples)])
        #<tensor>.index_select(0，j) 第一个参数为0表示按行所有 第二个参数为索引的序号
        yield  features.index_select(0, j), labels.index_select(0, j)

batch_size = 10
for X, y in data_iter(batch_size, features, labels):
    print(X, '\n', y)
    break


#定义模型
def linreg(X, w, b):
    return torch.mm(X, w) + b

#定义损失函数
def squared_loss(y_hat, y):
    return (y_hat - y.view(y_hat.size())) ** 2 / 2

#定义优化函数
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= lr * param.grad / batch_size # ues .data to operate param without gradient track

#初始化模型参数
w = torch.tensor(np.random.normal(0, 0.01, (num_inputs, 1)), dtype=torch.float32)
b = torch.zeros(1, dtype=torch.float32)
w.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)

#训练
# super parameters init
lr = 0.03
num_epochs = 5

net = linreg
loss = squared_loss

# training
# training repeats num_epochs times
for epoch in range(num_epochs):
    # in each epoch, all the samples in dataset will be used once
    # X is the feature and y is the label of a batch sample
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y).sum()
        # calculate the gradient of batch sample loss
        l.backward()
        # using small batch random gradient descent to iter model parameters
        sgd([w, b], lr, batch_size)
        # reset parameter gradient
        w.grad.data.zero_()
        b.grad.data.zero_()
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().item()))

#####################线性回归 从零开始实现#################################

#####################线性回简单实现#################################

from torch import nn
#初始化参数
from torch.nn import init

torch.manual_seed(1)

print(torch.__version__)
torch.set_default_tensor_type('torch.FloatTensor')
import torch.utils.data as Data
batch_size = 10

# combine featues and labels of dataset
dataset = Data.TensorDataset(features, labels)

# put dataset into DataLoader
data_iter = Data.DataLoader(
    dataset=dataset,            # torch TensorDataset format
    batch_size=batch_size,      # mini batch size
    shuffle=True,               # whether shuffle the data or not
    num_workers=2,              # read data in multithreading
)
for X, y in data_iter:
    print(X, '\n', y)
    break

#定义模型
class LinearNet(nn.Module):
    def __init__(self, n_feature):
        # call father function to init
        super(LinearNet, self).__init__()
        # function prototype: `torch.nn.Linear(in_features, out_features, bias=True)`
        self.linear = nn.Linear(n_feature,1)

    def forward(self, x):
        y = self.linear(x)
        return y


net = LinearNet(num_inputs)
print(net)

# ways to init a multilayer network
# method one
net = nn.Sequential(
    nn.Linear(num_inputs, 1)
    # other layers can be added here
    )

# method two
# net.add_module ......
net = nn.Sequential()
net.add_module('linear', nn.Linear(num_inputs, 1))

# method three
from collections import OrderedDict
net = nn.Sequential(OrderedDict([('linear', nn.Linear(num_inputs, 1))]))

print(net)
print(net[0])

init.normal_(net[0].weight, mean=0.0, std=0.01)
# or you can use `net[0].bias.data.fill_(0)` to modify it directly
init.constant_(net[0].bias, val=0.0)
for param in net.parameters():
    print(param)

#定义损失函数
 # nn built-in squared loss function
# function prototype: `torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')`
loss = nn.MSELoss()

#定义优化函数
import torch.optim as optim
# built-in random gradient descent function
# function prototype: `torch.optim.SGD(params, lr=, momentum=0, dampening=0, weight_decay=0, nesterov=False)`
optimizer = optim.SGD(net.parameters(), lr=0.03)
print(optimizer)

#训练
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        output = net(X)
        l = loss(output, y.view(-1, 1))
        # reset gradient, equal to net.zero_grad()
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
    print('epoch %d, loss: %f' % (epoch, l.item()))

# result comparision
dense = net[0]
print(true_w, dense.weight.data)
print(true_b, dense.bias.data)
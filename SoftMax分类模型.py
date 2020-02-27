# import needed package
from IPython import display
import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms
import time

import sys
import numpy as np
import d2lzh_pytorch as d2l

print(torch.__version__)
print(torchvision.__version__)


#####################Fashion Mnist数据#################################<<<<

#下载数据 训练／测试数据
mnist_train = torchvision.datasets.FashionMNIST(root='Data/softmax/FashionMNIST2065', train=True, download=True, transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='Data/softmax/FashionMNIST2065', train=False, download=True, transform=transforms.ToTensor())


# show result
print(type(mnist_train))
print(len(mnist_train), len(mnist_test))

# 可通过下标来访问任意一个样本
feature, label = mnist_train[0]
# Channel x Height x Width
print(feature.shape, label)
#查看图片类型
mnist_PIL = torchvision.datasets.FashionMNIST(root='Data/softmax/FashionMNIST2065', train=True, download=True)
PIL_feature, label = mnist_PIL[0]
print(PIL_feature)

# 函数保存在d2lzh包
def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # _表示忽略的变量
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.view((28, 28)).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

#显示图片
X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0]) # 将第i个feature加到X中
    y.append(mnist_train[i][1]) # 将第i个label加到y中
show_fashion_mnist(X, get_fashion_mnist_labels(y))

#####################Fashion Mnist数据#################################>>>

#对多维Tensor操作
X = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(X.sum(dim=0, keepdim=True))  # dim为0，按照相同的列求和，并在结果中保留列特征
print(X.sum(dim=1, keepdim=True))  # dim为1，按照相同的行求和，并在结果中保留行特征
print(X.sum(dim=0, keepdim=False)) # dim为0，按照相同的列求和，不在结果中保留列特征
print(X.sum(dim=1, keepdim=False)) # dim为1，按照相同的行求和，不在结果中保留行特征


# 读取数据
batch_size = 256
num_workers = 4
train_iter = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
start = time.time()
for X, y in train_iter:
    continue
print('%.2f sec' % (time.time() - start))


#####################softmax 从零开始实现#################################<<<
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='Data/softmax/FashionMNIST2065')
#模型参数初始化
num_inputs = 784
print(28*28)
num_outputs = 10
W = torch.tensor(np.random.normal(0, 0.01, (num_inputs, num_outputs)), dtype=torch.float)
b = torch.zeros(num_outputs, dtype=torch.float)
W.requires_grad_(requires_grad=True)
b.requires_grad_(requires_grad=True)
#定义softmax
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    # print("X size is ", X_exp.size())
    # print("partition size is ", partition, partition.size())
    return X_exp / partition  # 这里应用了广播机制
X = torch.rand((2, 5))
X_prob = softmax(X)
print(X_prob, '\n', X_prob.sum(dim=1))


#softmax回归模型

def net(X):
    return softmax(torch.mm(X.view((-1,num_inputs)),W)+b) #torch.mm() 为矩阵相乘

#定义损失函
y_hat= torch.Tensor([[0.1,0.3,0.6],[0.3, 0.2, 0.5]])
y = torch.LongTensor([0,2])
y_hat.gather(1,y.view(-1,1)) #gather 按列以y。view(-1,1)索引获取概率
def cross_entropy(y_hat,y):
    return -torch.log(y_hat.gather(1,y.view(-1,1)))

#定义准确率
def accuracy(y_hat,y):
    return (y_hat.argmax(dim=1)==y).float().mean().item()
print(accuracy(y_hat,y))

#该函数已存d2lzh
def evaluate_accuuracy(data_iter,net):
    acc_sum,n = 0.0,0
    for X,y in data_iter:
        acc_sum += (net(X).argmax(dim=1)== y).float().sum().item()
        n += y.shape[0]
    return acc_sum/n
print(evaluate_accuuracy(test_iter,net))

#训练模型
import torch.optim as optim
optimizer = optim.SGD(net.parameters(), lr=0.03)
num_epoch,lr = 5,0.1

def train_ch3(net,train_iter,test_iter,loss,num_epoch,batch_size,
              params=None,lr = None,optimizer =None):
    for epoch in range(num_epoch):
        train_loss_sum =0
        train_acc_sum =0
        n =0
        for X,y in train_iter:
            y_hat = net(X)
            l = loss(y_hat,y).sum().item()
            print(l)
            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0] is not None:
                for param in params:
                    param.grad.data.zero_()

            loss.backward()
            if optimizer is None:
                d2l.sgd(params,lr,batch_size)
            else:
                optimizer.step()

            train_loss_sum +=l
            train_acc_sum += (y_hat.argmax(dim=1)==y).sum().item()
            n +=y.shape[0]
        test_acc = evaluate_accuuracy(test_iter,net)
        print('epoch %d, loss %.4f,train acc %.3f,test %.3f'
        %(epoch+1,train_loss_sum/n,train_acc_sum/n,test_acc))

train_ch3(net, train_iter, test_iter, cross_entropy, num_epoch, batch_size, [W, b], lr,)

#模型预测
X, y = iter(test_iter).next()
true_labels = d2l.get_fashion_mnist_labels(y.numpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(dim=1).numpy())
titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]
d2l.show_fashion_mnist(X[0:9], titles[0:9])

#####################softmax 从零开始实现#################################>>>

##################### softmax 简单实现#################################>>>

num_inputs = 784
num_outputs =10
num_epochs = 5

from torch import nn
from collections import OrderedDict

class LinearNet(nn.Module):
    def __init__(self,num_inputs,num_outputs):
        super(LinearNet,self).__init__()
        self.liner = nn.Linear(num_inputs,num_outputs)
    def forward(self,x):#x 形状 (batch,1,28,28)
        return self.liner(x.view(x.shape[0],-1))


class FlattenLayer(nn.Module):
    def __init__(self):
        super(FlattenLayer, self).__init__()
    def forward(self, x):
        return x.view(x.shape[0],-1)

net = nn.Sequential(
    OrderedDict(
        [('flatten',FlattenLayer()),
         # ('linear',LinearNet()),
         ('linear',nn.Linear(num_inputs,num_outputs))]
    )
)

nn.init.normal(net.linear.weight,mean =0,std =0.01) #初始化模型参数
nn.init.constant_(net.linear.bias,val =0)
loss = nn.CrossEntropyLoss()#定义损失函数
optimizer = torch.optim.SGD(net.parameters(),lr =0)#定义优化函数 nei为Order
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)


##################### 多层感知 简单实现#################################>>>
num_inputs = 784
num_outputs =10
num_hiddens = 256

num_epochs = 5
batch_size = 256

net = nn.Sequential(
    OrderedDict([
        ('flatten',d2l.FlattenLayer()),
        ('hidden',nn.Linear(num_inputs,num_hiddens)),
        ('relu',nn.ReLU()),
        ('linear',nn.Linear(num_hiddens,num_outputs))
    ])
)
for param  in net.parameters():
    nn.init.normal_(param,mean=0,std =0.01)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.5)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)
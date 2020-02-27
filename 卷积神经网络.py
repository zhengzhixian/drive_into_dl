import torch
import torch.nn as nn

##################### 多层感知 从零实现#################################>>>

'''
1、二维卷积
二维卷积层将输入和卷积核做互相关运算，并加上一个标量偏置来得到输出。
卷积层的模型参数包括卷积核和标量偏置。
'''
def corr2d(X,K):
    H,W = X.shape
    h,w = K.shape
    Y = torch.zeros(H-h+1,W-w+1)
    for  i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i,j]= (X[i:i+h,j:j+w]*K).sum()
    return Y

'''
#test1
X = torch.tensor([[0,1,2],[3,4,5],[6,7,8]])
K = torch.tensor([[0,1],[2,3]])
Y= corr2d(X,K)
print(Y)

'''
class Conv2D(nn.Module):
    def __init__(self,kernel_size):
        super(Conv2D,self).__init__() #继承自父类的属性进行初始化
        self.weight = nn.Parameter(torch.randn(kernel_size))
        self.bias = nn.Parameter(torch.randn(1))
    def forward(self, x):
        return corr2d(x,self.weight)+self.bias


###### 学习 1x2卷积层 卷积层检测颜色边缘
X = torch.ones(6, 8)
Y = torch.zeros(6, 7)
X[:, 2: 6] = 0
Y[:, 1] = 1
Y[:, 5] = -1

con2d = Conv2D(kernel_size=(1,2))
step = 30
lr = 0.01
for i in range(step):
    Y_hat= con2d.forward(X)
    l= ((Y - Y_hat)**2).sum()
    l.backward()  #反向传播
    con2d.weight.data -= lr*con2d.weight.grad  #optim 优化参数weight
    con2d.bias.data -= lr*con2d.bias.grad  #optim 优化参数bias
    con2d.weight.grad.zero_()     #梯度清零
    con2d.bias.grad.zero_()
    if i%10 == 0:
        print('Step %d, loss %.3f' % (i + 1, l.item()))

print(con2d.weight.data)
print(con2d.bias.data)



##################### 多层感知 简单实现#################################>>>

'''
一、卷积层
1、多输出通道
卷积层的输出也可以包含多个通道，设卷积核输入通道数和输出通道数分别为 𝑐𝑖 和 𝑐𝑜 ，高和宽分别为 𝑘ℎ 和 𝑘𝑤 。
如果希望得到含多个通道的输出，我们可以为每个输出通道分别创建形状为 𝑐𝑖 × 𝑘ℎ × 𝑘𝑤 的核数组，
将它们在输出通道维上连结，卷积核的形状即 𝑐𝑜 × 𝑐𝑖 × 𝑘ℎ × 𝑘𝑤 。

对于输出通道的卷积核，一个 𝑐𝑖 × 𝑘ℎ × 𝑘𝑤 的核数组可以提取某种局部特征，
但是输入可能具有相当丰富的特征，需要有多个 𝑐𝑖 × 𝑘ℎ × 𝑘𝑤的核数组，不同的核数组提取的是不同的特征。

2、1x1卷积层

输入通道3 输出通道2 的1x1。
1×1 卷积核可在不改变高宽的情况下，调整通道数。 
1×1 卷积核不识别高和宽维度上相邻元素构成的模式，其主要计算发生在通道维上。
假设我们将通道维当作特征维，将高和宽维度上的元素当成数据样本，那么 1×1 卷积层的作用与全连接层等价。

3、简单实现
使用Pytorch中的nn.Conv2d类来实现二维卷积层，主要关注以下几个构造函数参数：
in_channels (python:int) – Number of channels in the input imag
out_channels (python:int) – Number of channels produced by the convolution
kernel_size (python:int or tuple) – Size of the convolving kernel
stride (python:int or tuple, optional) – Stride of the convolution. Default: 1
padding (python:int or tuple, optional) – Zero-padding added to both sides of the input. Default: 0
bias (bool, optional) – If True, adds a learnable bias to the output. Default: True
forward函数的参数为一个四维张量，形状为 (𝑁,𝐶𝑖𝑛,𝐻𝑖𝑛,𝑊𝑖𝑛) ，
返回值也是一个四维张量，形状为 (𝑁,𝐶𝑜𝑢𝑡,𝐻𝑜𝑢𝑡,𝑊𝑜𝑢𝑡) ，
中 𝑁 是批量大小， 𝐶,𝐻,𝑊 分别表示通道数、高度、宽度。
'''

X = torch.randn(4,2,3,5)
conv2d1 = nn.Conv2d(in_channels=2, out_channels=3, kernel_size=(3, 5), stride=1, padding=(1, 2))
Y = conv2d1(X)
print('Y.shape: ', Y.shape)
print('weight.shape: ', conv2d1.weight.shape)
print('bias.shape: ', conv2d1.bias.shape)

'''
二 池化层
1、池化层可以在输入的高和宽两侧填充并调整窗口的移动步幅来改变输出形状。
池化层填充和步幅与卷积层填充和步幅的工作机制一样。

在处理多通道输入数据时，池化层对每个输入通道分别池化，
不会像卷积层那样将各通道的结果按通道相加。这意味着池化层的输出通道数与输入通道数相等。

2、简单实现
Pytorch中的nn.MaxPool2d实现最大池化层，关注以下构造函数参数：
kernel_size – the size of the window to take a max over
stride – the stride of the window. Default value is kernel_size
padding – implicit zero padding to be added on both sides
forward函数的参数为一个四维张量，形状为 (𝑁,𝐶,𝐻𝑖𝑛,𝑊𝑖𝑛) ，返回值也是一个四维张量，
形状为 (𝑁,𝐶,𝐻𝑜𝑢𝑡,𝑊𝑜𝑢𝑡) ，其中 𝑁 是批量大小， 𝐶,𝐻,𝑊 分别表示通道数、高度、宽度。
平均池化层使用的是nn.AvgPool2d，使用方法与nn.MaxPool2d相同
'''
X = torch.arange(32, dtype=torch.float32).view(1, 2, 4, 4)
pool2d = nn.MaxPool2d(kernel_size=3, padding=1, stride=(2, 1))
Y = pool2d(X)
print(X)
print(Y)
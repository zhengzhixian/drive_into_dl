'''
LeNet
'''

import torch
import torch.nn as nn
import d2lzh_pytorch as d2l
import matplotlib.pyplot as plt
import torch.optim as optim
import  time

#######################LeNet##################

'''
LeNet分为卷积层块和全连接层块两个部分。
卷积层块里的基本单位是卷积层后接平均池化层：卷积层用来识别图像里的空间模式，如线条和物体局部，之后的平均池化层则用来降低卷积层对位置的敏感性。
卷积层块由两个这样的基本单位重复堆叠构成。在卷积层块中，每个卷积层都使用 5×5 的窗口，并在输出上使用sigmoid激活函数。
第一个卷积层输出通道数为6，第二个卷积层输出通道数则增加到16。
全连接层块含3个全连接层。它们的输出个数分别是120、84和10，其中10为输出的类别个数。

以下可看出
在卷积层块中输入的高和宽在逐层减小。卷积层由于使用高和宽均为5的卷积核，
从而将高和宽分别减小4，而池化层则将高和宽减半，但通道数则从1增加到16。全连接层则逐层减少输出个数，直到变成图像的类别数10。
'''


class Faltten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0],-1)

class Reshape(nn.Module):
    def forward(self, x):
        return x.view(-1,1,28,28)   #(B H H W)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv = nn.Sequential(
                Reshape(),
                nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5,padding=2),# bx1x28x28 bx6x28*28
                nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2,stride=2), #bx6x28*28 -> bx6x14x14

                nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),# bx1x14x14 -> bx16x10*10
                nn.Sigmoid(),
                nn.AvgPool2d(kernel_size=2,stride=2), #bx16x10*10 -> bx16x5*5
        )
        #2卷积层和 3层全链接 需进行Flatten
        self.fc = nn.Sequential(
                Faltten(),
                nn.Linear(in_features=16 * 5 * 5, out_features=120),
                nn.Sigmoid(),
                nn.Linear(in_features=120, out_features=84),
                nn.Sigmoid(),
                nn.Linear(84, 10)
        )
    def forward(self, X):
        feature=self.conv(X)
        output = self.fc(feature)
        return output

net = torch.nn.Sequential(     #Lelet
    Reshape(),
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2), #b*1*28*28  =>b*6*28*28
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),                              #b*6*28*28  =>b*6*14*14
    nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),           #b*6*14*14  =>b*16*10*10
    nn.Sigmoid(),
    nn.AvgPool2d(kernel_size=2, stride=2),                              #b*16*10*10  => b*16*5*5
    d2l.FlattenLayer(),                                                          #b*16*5*5   => b*400
    nn.Linear(in_features=16*5*5, out_features=120),
    nn.Sigmoid(),
    nn.Linear(120, 84),
    nn.Sigmoid(),
    nn.Linear(84, 10)
)

LeNet = LeNet()
print(LeNet)

####################### AlexNet ##################

'''
LeNet: 在大的真实数据集上的表现并不尽如⼈意。
1.神经网络计算复杂。
2.还没有⼤量深⼊研究参数初始化和⾮凸优化算法等诸多领域。

AlexNet
首次证明了学习到的特征可以超越⼿⼯设计的特征，从而⼀举打破计算机视觉研究的前状。
特征：
8层变换，其中有5层卷积和2层全连接隐藏层，以及1个全连接输出层。
将sigmoid激活函数改成了更加简单的ReLU激活函数。
用Dropout来控制全连接层的模型复杂度。
引入数据增强，如翻转、裁剪和颜色变化，从而进一步扩大数据集来缓解过拟合。

CPU训练较慢 gpu运行移至kaggle
https://www.kaggle.com/boyuai/boyu-d2l-modernconvolutionalnetwork
'''

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.conv = nn.Sequential(
            # 3 x 224 x 224
            nn.Conv2d(in_channels=1,out_channels=96,kernel_size=11,stride=4), # 96*54*54
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),#96*26*26
            #减小卷积窗口 padding=2使得输入输出高宽一致 且增大输出通道
            nn.Conv2d(in_channels=96,out_channels=256,kernel_size=5,padding=2),#256*26*26
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),#256*12*12
            nn.Conv2d(in_channels=256,out_channels=384,kernel_size=3,padding=1),#384*12*12
            nn.Conv2d(in_channels=384,out_channels=384,kernel_size=3,padding=1),#384*12*12
            nn.Conv2d(in_channels=384,out_channels=256, kernel_size=3,padding=1),#384*12*12
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),#256*5*5
        )
        #丢弃部分层缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5,10),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096,10)
        )

    def forward(self, X):
        feature = self.conv(X)
        output = self.fc(feature.view(X.shape[0],-1))
        return output

AlexNet = AlexNet()
print(AlexNet)

####################### VGG ##################
'''
VGG：通过重复使⽤简单的基础块来构建深度模型。
Block:数个相同的填充为1、窗口形状为 3×3 的卷积层,接上一个步幅为2、窗口形状为 2×2 的最大池化层。
卷积层保持输入的高和宽不变，而池化层则对其减半。
'''
class VGG(nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        # conv_arch 有5个vgg block，每个block表示(卷积数，输入通道，输出通道)
        self.conv_arch = ((1, 1, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        self.fc_features = 512 * 7 * 7  # 经过5个conv 狂高减半5次 图片大小变为224／32 = 7
        self.fc_hidden_unit = 4096

        self.net = nn.Sequential()
        #卷积
        for i, (num_convs,in_channels, out_channels) in enumerate(self.conv_arch):
            self.net.add_module('vgg_block_'+str(i+1),self.vgg_block(num_convs,in_channels,out_channels))
        #全链接
        self.net.add_module('fc',nn.Sequential(
            d2l.FlattenLayer(),
            nn.Linear(self.fc_features,self.fc_hidden_unit),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_hidden_unit,self.fc_hidden_unit),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(self.fc_hidden_unit,10)
        ))

    def vgg_block(self,nums_conv,in_channels,out_channels):
        blk = []
        for i in range(nums_conv):
            if i==0:
                blk.append(nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1))
            else:
                blk.append(nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1))
            blk.append(nn.ReLU())
        blk.append(nn.MaxPool2d(kernel_size=2,stride=2)) #宽高减半
        return nn.Sequential(*blk)

    def forward(self, X):
        return self.net(X)


vgg = VGG()

def getLayOfshape(net):
    #查看每层的输入输出形状大小
    X = torch.randn(1,1,224,224)
    #named_children获取一级子模块及其名字
    #named_modules会返回所有子模块,包括子模块的子模块
    for name,blk in net.named_children():
        print(name,blk)
        X = blk(X)
        print(name,'ouput shape: ',X.shape)

getLayOfshape(vgg)

####################### NiN ##################
'''
LeNet、AlexNet和VGG：先以由卷积层构成的模块充分抽取 空间特征，再以由全连接层构成的模块来输出分类结果。
NiN：串联多个由卷积层和“全连接”层构成的小⽹络来构建⼀个深层⽹络。
⽤了输出通道数等于标签类别数的NiN块，然后使⽤全局平均池化层对每个通道中所有元素求平均并直接⽤于分类。

1×1卷积核作用
1.放缩通道数：通过控制卷积核的数量达到通道数的放缩。
2.增加非线性。1×1卷积核的卷积过程相当于全连接层的计算过程，并且还加入了非线性激活函数，从而可以增加网络的非线性。
3.计算参数少

Nin由1x1卷积（替代全链接层）构成的NIN块构建深度网络
NIN去出了全链接层，替换为 输出通道=标签类别的NIN块 + 全局平均池化
'''
def nin_block(in_channels, out_channels, kernel_size, stride, padding):
    blk = nn.Sequential(
        nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1),
        nn.ReLU())
    return blk

nin_net = nn.Sequential(
    nin_block(in_channels=1,out_channels=96,kernel_size=11,stride=4,padding=0),#96*54*54
    nn.MaxPool2d(kernel_size=3,stride=2),#96*26*26
    nin_block(in_channels=96,out_channels=256,kernel_size=5,stride=1,padding=2),#256*26*26
    nn.MaxPool2d(kernel_size=3,stride=2),#256*12*12
    nin_block(in_channels=256,out_channels=384,kernel_size=3,stride=1,padding=1),#384*12*12
    nn.MaxPool2d(kernel_size=3,stride=2),#384*5*5
    nn.Dropout(0.5),
    nin_block(in_channels=384, out_channels=10, kernel_size=3, stride=1,padding=1),#10*5*5
    d2l.GlobalAvgPool2d(),#10*1+1
    d2l.FlattenLayer()#10
)
X = torch.rand(1, 1, 224, 224)
for name,l in nin_net.named_children():
    x = l(X)
    print(name,'out shape: ',x.shape)

####################### GoogLeNet ##################
'''
1、由Inception基础块组成。
2、Inception块相当于⼀个有4条线路的⼦⽹络。它通过不同窗口形状的卷积层和最⼤池化层来并⾏抽取信息，并使⽤1×1卷积层减少通道数从而降低模型复杂度。
3、可以⾃定义的超参数是每个层的输出通道数，我们以此来控制模型复杂度。
'''

class Inception(nn.Module):
    def __init__(self,in_c,c1,c2,c3,c4):
        super(Inception,self).__init__() #继承父类初始化方法属性
        self.p1_1 = nn.Conv2d(in_c,c1,kernel_size=1)
        #线路2 1*1卷积 -> 3*3卷积
        self.p2_1 = nn.Conv2d(in_c,c2[0],kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0],c2[1],kernel_size=3,padding=1) #图像大小不变
        #线路3 1*1卷积 -> 3*3卷积
        self.p3_1 = nn.Conv2d(in_c,c3[0],kernel_size=1)
        self.p3_2= nn.Conv2d(c3[0],c3[1],kernel_size=5,padding=2),#图像大小不变
        self.p4_1 = nn.MaxPool2d(kernel_size=3,padding=1),
        self.p4_2 = nn.Conv2d(in_c,c4,kernel_size=1)


    def forward(self, x):
        p1 = nn.Sequential(self.p1_1,
                           nn.ReLU())

        p2 = nn.Sequential(self.p2_1,
                           nn.ReLU(),
                           self.p2_2,
                           nn.ReLU())

        p3 = nn.Sequential(self.p3_1,
                           nn.ReLU(),
                           self.p3_2,
                           nn.ReLU())

        p4 = nn.Sequential(self.p4_1,
                           self.p4_2,
                           nn.ReLU())
        return torch.cat([p1,p2,p3,p4],1)


#Google net 完整结构
b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                   nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                   nn.Conv2d(64, 192, kernel_size=3, padding=1),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b3 = nn.Sequential(Inception(192, 64, (96, 128), (16, 32), 32),
                   Inception(256, 128, (128, 192), (32, 96), 64),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b4 = nn.Sequential(Inception(480, 192, (96, 208), (16, 48), 64),
                   Inception(512, 160, (112, 224), (24, 64), 64),
                   Inception(512, 128, (128, 256), (24, 64), 64),
                   Inception(512, 112, (144, 288), (32, 64), 64),
                   Inception(528, 256, (160, 320), (32, 128), 128),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

b5 = nn.Sequential(Inception(832, 256, (160, 320), (32, 128), 128),
                   Inception(832, 384, (192, 384), (48, 128), 128),
                   d2l.GlobalAvgPool2d())

net = nn.Sequential(b1, b2, b3, b4, b5,
                    d2l.FlattenLayer(),
                    nn.Linear(1024, 10))

net = nn.Sequential(b1, b2, b3, b4, b5, d2l.FlattenLayer(), nn.Linear(1024, 10))

X = torch.rand(1, 1, 96, 96)
for blk in net.children():
    X = blk(X)
    print('output shape: ', X.shape)


####################### data Explore  ##################

net = LeNet
#构造样本 查看每层含义及输出
X = torch.randn(size=(1,1,28,28), dtype = torch.float32)
for layer in net:
    X = layer(X)
    print(layer.__class__.__name__,'output shape: \t',X.shape)

#加载 fashion_mnist
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='Data/softmax/FashionMNIST2065')

#画图 批量获取画图
def show_fashion_mnist(images,labels):
    d2l.use_svg_display()
    _,figs = plt.subplots(1,len(images),figsize =(12,12))
    for f, img,lbl in zip(figs,images,labels):
        f.imshow(img.view(28,28).numpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
    plt.show()

#一个batch内偶去10个样本
for Xdata,ylabel in train_iter:
    break
X,y =[],[]
for i in range(10):
    X.append(Xdata[i])
    y.append(ylabel[i].numpy())

show_fashion_mnist(X,y)

#积神经网络计算比多层感知机要复杂，建议使用GPU来加速计算。
# 我们查看看是否可以用GPU，如果成功则使用cuda:0，否则仍然使用cpu。

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def try_gpu():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    return device
device = try_gpu()


####################### Train  ##################

####################### Train1  ##################

num_epochs =10
loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(),lr = 0.5)
# optimizer = torch.optim.Adam(net.parameters(),lr=0.5)
d2l.train_ch3(net,train_iter,test_iter,loss,num_epochs,batch_size,None,None,optimizer)

####################### Train2  ##################
def train_ch5(net, train_iter, test_iter, criterion, num_epochs, batch_size, device, lr=None):
    """Train and evaluate a model with CPU or GPU."""
    print('training on', device)
    net.to(device)
    optimizer = optim.SGD(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_l_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        train_acc_sum = torch.tensor([0.0], dtype=torch.float32, device=device)
        n, start = 0, time.time()
        for X, y in train_iter:
            net.train()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                y = y.long()
                train_l_sum += loss.float()
                train_acc_sum += (torch.sum((torch.argmax(y_hat, dim=1) == y))).float()
                n += y.shape[0]
        test_acc = d2l.evaluate_accuracy(test_iter, net, device)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,

                 time.time() - start))

lr, num_epochs = 0.9, 10
def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform_(m.weight)

net.apply(init_weights)
net = net.to(device)
criterion = nn.CrossEntropyLoss()   #交叉熵描述了两个概率分布之间的距离，交叉熵越小说明两者之间越接近
train_ch5(net, train_iter, test_iter, criterion,num_epochs, batch_size,device, lr)



'''
批归一化
'''
'''
对输入的标准化（浅层模型）
处理后的任意一个特征在数据集中所有样本上的均值为0、标准差为1。
标准化处理输入数据使各个特征的分布相近

批量归一化（深度模型）
利用小批量上的值和标准差，不断调整神经网络中间输出，从而使整个神经网络在各层的中间输出的数值更稳定。
1、对全链接做批归一化
位置：全链接层与激活函数之间
全连接：𝑥=𝑊𝑢+𝑏   𝑜𝑢𝑡𝑝𝑢𝑡=𝜙(𝑥)
批量归一化：可学习参数 拉伸参数 和偏移参数 

2.对卷积层做批量归⼀化
位置：卷积计算之后、应⽤激活函数之前。
如果卷积计算输出多个通道，需要对这些通道的输出分别做批量归一化，且每个通道都拥有独立的拉伸和偏移参数。 
计算：对单通道，batchsize=m,卷积计算输出=pxq 对该通道中m×p×q个元素同时做批量归一化,使用相同的均值和方差。

3.预测时的批量归⼀化
训练：以batch为单位,对每个batch计算均值和方差。
预测：用移动平均估算整个训练数据集的样本均值和方差。                    
'''

import torch
import torch.nn as nn
import d2lzh_pytorch as d2l
import torch.nn.functional as F

####################### 批量归一化 ##################

def batch_norm(is_trainning,X,gamma,beta,moving_mean,moving_var,eps,monmentum):
    if not is_trainning:
        X_hat = (X-moving_mean)/torch.sqrt(moving_var+eps) #预测则直接驶入移动平均和方差
    else:
        assert len(X.shape) in [2,4] #全🔗 len(X.shape) = 2 卷积：len(X.shap)e=4
        if (X.shape==2):
            mean = X.mean(dim =0)
            var = ((X - mean)**2).mean(dim=0)
        else:
            mean  = X.mean(dim=0,keepdim = True).mean(dim=2,keepdim = True).mean(dim=3,keepdim = True)
            var = ((X - mean)**2).mean(dim=0,keepdim = True).mean(dim=2,keepdim = True).mean(dim =3,keepdim = True)
        X_hat = (X -mean)/torch.sqrt(var+eps)
        #更新移动平均的均值和方差
        moving_mean = monmentum*moving_mean+(1-monmentum)*mean
        moving_var = monmentum*var +(1-monmentum)*var
    Y = gamma* X_hat +beta #拉升和偏移
    return Y,moving_mean,moving_var

class BatchNorm(nn.Module):
    def __init__(self,num_features,num_dims):
        super(BatchNorm,self).__init__()
        #nums_feature BN的独立维度
        if num_dims ==2:
            shape = (1,num_features) #全链接输出神经元
        else:
            shape =(1,num_features,1,1) #通道数
        #拉伸 偏移参与梯度计算 初始化0 1
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.zeros(shape)

    def forward(self,X):
        # 若X不在内存 将moving_mean moving_var 复制到X所在的显存
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        Y ,self.moving_mean,self.moving_var = \
            batch_norm(self.training,X,self.gamma,self.beta,self.moving_mean,self.moving_var,
                   eps=1e-5,monmentum=0.9)
        return Y

#基于LeNet应用
net = nn.Sequential(
    nn.Conv2d(in_channels=1,out_channels=6,kernel_size=5),
    BatchNorm(6,num_dims =4),
    #nn.BatchNorm2d(6)
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2,padding=2),
    nn.Conv2d(in_channels=6,out_channels=16,kernel_size=5),
    BatchNorm(16,num_dims=4),
    #nn.BatchNorm2d(16),
    nn.Sigmoid(),
    nn.MaxPool2d(kernel_size=2,padding=2),
    d2l.FlattenLayer(),
    nn.Linear(16*4*4,120),
    BatchNorm(120,num_dims=2),
    # nn.BatchNorm1d(120)
    nn.Sigmoid(),
    nn.Linear(120,84),
    BatchNorm(84,num_dims=2),
    # nn.BatchNorm1d(84)
    nn.Sigmoid(),
    nn.Linear(84,10)
)

print(net)

torch.optim.Adam(net.parameters(),lr = 0.1)

####################### ResNet ##################
'''
设定输出通道，是否使用额外的1x1卷积层修改通道数
若 residual 中in_channel 和out_channel相等 则不使用1x1卷积改变通道 直接与输入的Tensor X 相加
若不同 则使用1x1卷积 改变通道
Residual 可直接在d2l中调用
'''
#构建餐叉网络
class Residual(nn.Module):
    def __init__(self,in_channels,out_channels,use_1x1conv =False,stride =1):
        super(Residual,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size=3,padding=1,stride=stride)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, X):
        X1 = F.relu(self.bn1(self.conv1(X)))
        Y= self.bn2((self.conv2(X1)))
        #1x1 卷积输改变通道数
        if self.conv3:
             X = self.conv3(X) + Y
        #shortcut 相加
        return F.relu(X + Y)

#in_channel = out_channel
X = torch.randn((4,3,6,6))
Res_blk1 = Residual(3,3) #产生对象
print(Res_blk1(X).shape) #torch.Size(4,3,6,6)

#in_channel != out_channel
Res_blk2 = Residual(3,6,use_1x1conv=True,stride=2)
print(Res_blk2(X).shape) #torch.Size(4,6,3,3)

'''
卷积(64,7x7,3)
批量一体化
最大池化(3x3,2)
残差块 x4 (步幅为2的残差块 之间减小高和宽)
全局平均池化
全连接
'''

def ResNet():
    net = nn.Sequential(
        nn.Conv2d(1,64,kernel_size=7,stride=2,padding=3),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2,stride=2,padding=1)
    )

    #堆砌残差网络
    def resnet_block(in_channels,out_channels,num_residuals,first_block=False):
        if first_block:
            assert in_channels == out_channels
        blk = []
        for i in range(num_residuals):
            if i==0 and not first_block:
                blk.append(Residual(in_channels,out_channels,use_1x1conv=True,stride=2))
            else:
                blk.append(Residual(out_channels,out_channels))
        return nn.Sequential(*blk)

    net.add_module('resnet_block1',resnet_block(64,64,2,first_block=True))
    net.add_module('resnet_block2',resnet_block(64,128,2)) #通道增加 图像大小减半
    net.add_module('resnet_block3',resnet_block(128,256,2))
    net.add_module('resnet_block4',resnet_block(256,512,2))

    net.add_module('gloval_ave_pool',d2l.GlobalAvgPool2d()) #输出 (batch,512,1,1)
    net.add_module('fc',nn.Sequential(d2l.FlattenLayer(),
                                      nn.Linear(512,10)))

    return net

Rsnet = ResNet()
X = torch.rand(1,1,224,224)
for name,layer in Rsnet.named_children():
    X = layer(X)
    print(name,' output shape: ',X.shape)

####################### 稠密连接网络(DenseNet) ##################

#稠密块(dense block) 定义了输入和输出如何连结
#过度层(transition layer) 用来控制通道数
#ResNet在跨层连接相加 与 DenseNet 跨层连接连接

class DenseBlock(nn.Module):
    def __init__(self,num_convs,in_channels,out_channels):
        super(DenseBlock,self).__init__()
        net = []
        for i in range(num_convs):
             in_c = in_channels + i*out_channels
             net.append(self.conv_block(in_c,out_channels))
        self.net = nn.ModuleList(net)
        self.out_channels = in_channels + num_convs*out_channels

    #定义卷积层
    def conv_block(self,in_channels, out_channels):
        blk = nn.Sequential(nn.BatchNorm2d(in_channels),
                            nn.ReLU(),
                            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        return blk

    def forward(self, X):
        for blk in self.net:
            Y =  blk(X) #每一个blk输入通道数递增
            X = torch.cat([X,Y],dim = 1) #通道维度上将输入和输出连接
        return X

blk = DenseBlock(2,3,10)
X = torch.rand(4,3,8,8)
Y = blk(X)
print(Y.shape) # torch.Size([4,23,8,8])

#过度层
# 1x1卷积：减少通道数
# 步幅为2的平均池化层：减半高和宽
def transition_block(in_channels,out_channels):
    blk = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ReLU(),
        nn.Conv2d(in_channels,out_channels,kernel_size=1),
        nn.AvgPool2d(kernel_size=2,stride=2)
    )
    return blk


blk = transition_block(23,10)
print(blk(Y).shape) # torch.Size([4, 10, 4, 4])

#DenseNet 模型 稠密层+过渡层
def DenseNet():
    #初始模块
    net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

    num_channels,growth_rate = 64,32
    num_convs_in_dense_blocks = [4,4,4,4]
    # 稠密块 + 过渡块
    for i,num_convs in enumerate(num_convs_in_dense_blocks):
            DenseBlk= DenseBlock(num_convs,num_channels,growth_rate)
            net.add_module('DenseBlock_%d'%i,DenseBlk)
            # 稠密块的输出通道作为过渡层的输入
            num_channels = DenseBlk.out_channels
            #在稠密块(通道数增加) 之间加入过度层(图像大小减半，通道数减半)
            if i!=len(num_convs_in_dense_blocks) -1:
                TransBlk= transition_block(num_channels,num_channels//2)
                net.add_module('Trasition_block_%d'%i,TransBlk)
                num_channels = num_channels//2

    net.add_module('BN',nn.BatchNorm2d(num_channels))
    net.add_module('relu',nn.ReLU())
    #GlobalAvgPool2d 输出 (Batch,num_channels,1,1)
    net.add_module('global_avg_pool',d2l.GlobalAvgPool2d())
    net.add_module('fc',nn.Sequential(d2l.FlattenLayer(),
                                      nn.Linear(num_channels,10)))
    return net


def queryLayer(net):
    X= torch.rand(1,1,96,96)
    for name, layer in net.named_children():
        X = layer(X)
        print(name,'output shape:\t',X.shape)


DenseN = DenseNet()
queryLayer(DenseN)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 16
    # 如出现“out of memory”的报错信息，可减小batch_size或resize
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, root='Data/softmax/FashionMNIST2065')
    lr, num_epochs = 0.001, 5
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    d2l.train_ch5(net, train_iter, test_iter, batch_size, optimizer, device, num_epochs)
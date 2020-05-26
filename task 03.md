## Task 03 字符识别模型

前面的学习分别讲解了赛题理解和数据读取，这一节的主要内容是讲解常用的图像识别模型CNN，以及利用pytorch从头构建一个字符识别模型。

## 学习目标

- 了解CNN基础和原理
- 利用pytorch框架从头搭建CNN模型和利用已有模型

## CNN介绍

### Let-5模型

CNN模型是现在图像识别、计算机视觉领域的主流模型，也推动了本次人工智能浪潮的复兴，通过减少全连接层大幅度减少训练参数个数,同时支持网络深度化。CNN对输入的原始图片进行卷积、池化、全连接层对像素点进行缩减，减少图片尺寸，完成特征识别。

 ![IMG](task 03\卷积.png)


最经典的CNN模型Ｌｅｔ－５包含两层卷积、两层池化和两层全连接，最后一层全连接得到具体的分类输出,然后再与真实标签进行比较,将误差反向传播更新各层参数,更新完成后再次向前传播,直到训练完成.网络结构如下图所示:

 ![IMG](task 03\Le_CNN.png)

### 其他CNN模型

随着研究的深入,网络层次越来越深 训练参数越来越多的模型层出不穷,精度也优于最初的Let-5模型,典型的包括:

- AlexNet

   ![IMG](task 03\Alex-net.png)

- VGG-16

   ![IMG](task 03\VGG.png)

- Inception-v1

   ![IMG](task 03\Incep-net.png)

- ResNet-50

   ![IMG](task 03\Resnet50.png)

## 利用pytorch搭建字符识别模型

### 构建一个简单的字符识别模型

在Pytorch中构建模型,只需要定义模型参数和正向传播函数,Pytorch本身会自动计算反向传播.下面的代码构建了一个具有两个卷积层和六个全连接层并联的CNN模型．

```
import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset

# 定义模型
class SVHN_Model1(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
        # CNN提取特征模块
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(),  
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2)),
            nn.ReLU(), 
            nn.MaxPool2d(2),
        )
        # 
        self.fc1 = nn.Linear(32*3*7, 11)
        self.fc2 = nn.Linear(32*3*7, 11)
        self.fc3 = nn.Linear(32*3*7, 11)
        self.fc4 = nn.Linear(32*3*7, 11)
        self.fc5 = nn.Linear(32*3*7, 11)
        self.fc6 = nn.Linear(32*3*7, 11)
    
    def forward(self, img):        
        feat = self.cnn(img)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        c6 = self.fc6(feat)
        return c1, c2, c3, c4, c5, c6
    
model = SVHN_Model1()
```

接下来是训练代码:

```
# 损失函数
criterion = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.Adam(model.parameters(), 0.005)

loss_plot, c0_plot = [], []
# 迭代10个Epoch
for epoch in range(10):
    for data in train_loader:
        c0, c1, c2, c3, c4, c5 = model(data[0])
        loss = criterion(c0, data[1][:, 0]) + \
                criterion(c1, data[1][:, 1]) + \
                criterion(c2, data[1][:, 2]) + \
                criterion(c3, data[1][:, 3]) + \
                criterion(c4, data[1][:, 4]) + \
                criterion(c5, data[1][:, 5])
        loss /= 6
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss_plot.append(loss.item())
        c0_plot.append((c0.argmax(1) == data[1][:, 0]).sum().item()*1.0 / c0.shape[0])
        
    print(epoch)
```

### 利用预模型设置神经网络

使用在ImageNet数据集上的预训练模型resnet18,具体代码如下:

```
class SVHN_Model2(nn.Module):
    def __init__(self):
        super(SVHN_Model1, self).__init__()
                
        model_conv = models.resnet18(pretrained=True)
        model_conv.avgpool = nn.AdaptiveAvgPool2d(1)
        model_conv = nn.Sequential(*list(model_conv.children())[:-1])
        self.cnn = model_conv
        
        self.fc1 = nn.Linear(512, 11)
        self.fc2 = nn.Linear(512, 11)
        self.fc3 = nn.Linear(512, 11)
        self.fc4 = nn.Linear(512, 11)
        self.fc5 = nn.Linear(512, 11)
    
    def forward(self, img):        
        feat = self.cnn(img)
        # print(feat.shape)
        feat = feat.view(feat.shape[0], -1)
        c1 = self.fc1(feat)
        c2 = self.fc2(feat)
        c3 = self.fc3(feat)
        c4 = self.fc4(feat)
        c5 = self.fc5(feat)
        return c1, c2, c3, c4, c5
```

## 本章小结

- CNN原理及典型模型介绍
- pytorch搭建CNN字符识别模型

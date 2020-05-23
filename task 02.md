# TASK 02 数据读取及扩增

## 一、本节任务

继上节完成赛题基本理解和baseline基本框架运行之后，教程开始从头梳理代码思路。这节主要是讲解如何利用python库读取并处理图像，以及利用pytorch完成数据读取和扩增。

- 学习目标
  - 了解pillow和opencv的基本命令完成图像读取
  - 学会使用dataset和dataloader完成图像信息数据读取及扩增

## 二、图像读取

由于赛题需要对图片内容进行字符识别，因此势必需要对图像本身进行读取，这里介绍两个常用的方法。

### Pillow

Pillow是Python图像处理函式库(PIL）的一个分支。Pillow提供了常见的图像读取和处理的操作，而且可以与**ipython notebook**无缝集成，是应用比较广泛的库。

pillow官网：**https://pillow.readthedocs.io/en/stable/**

放一段pillow的基本命令代码

```
from PIL import Image,ImageFilter #导入pillow库
im =Image.open(cat.jpg’) #读取图片
im2 = im.filter(ImageFilter.BLUR) #使用模糊命令
im2.save(‘blur.jpg’, ‘jpeg’)
```

### Opencv

Opencv是一个跨平台的图像处理库，功能十分强大。

OpenCV官网：**https://opencv.org/**
OpenCV Github：**https://github.com/opencv/opencv**
OpenCV 扩展算法库：**https://github.com/opencv/opencv_contrib**

在给出的代码中，使用的命令是pillow库的读取命令，比较简单，因此在此不做过多赘述，本人也没有就图像读取方面作过多思考。

## 三、图像数据扩增

数据扩增是深度学习中非常重要的一步，也是保证模型参数训练效果的关键步骤，可以增加训练样本数，有效缓解过拟合问题，增强模型的泛化能力。

通常来说深度学习模型的参数可以达到万级甚至百万级，而训练的样本数很难获得这么大的数量，因此合理的数据扩增是必要的。另外，数据扩增可以有效扩大样本空间，避免模型陷入图像某一局部特征。

对于图像数据来说，可以从颜色空间、尺度空间、样本空间等角度入手进行扩增，当然需要考虑具体目标选择不同种类的扩增，比如在本赛题中是对数字进行字符识别，翻转扩增会改变字符代表的含义，因此并不适用（如6翻转后变为9，标签发生改变）。

#### 常见的数据扩增方法（torchvision）

- transforms.CenterCrop 对图片中心进行裁剪
- transforms.ColorJitter 对图像颜色的对比度、饱和度和零度进行变换
- transforms.FiveCrop 对图像四个角和中心进行裁剪得到五分图像
- transforms.Grayscale 对图像进行灰度变换
- transforms.Pad 使用固定值进行像素填充
- transforms.RandomAffine 随机仿射变换
- transforms.RandomCrop 随机区域裁剪
- transforms.RandomHorizontalFlip 随机水平翻转
- transforms.RandomRotation 随机旋转
- transforms.RandomVerticalFlip 随机垂直翻转

#### 常用第三方数据扩增库

- #### torchvision

  官网：**https://github.com/pytorch/vision**
  pytorch官方提供的数据扩增库，提供了基本的数据数据扩增方法，可以无缝与torch进行集成；但数据扩增方法种类较少，且速度中等；

- #### imgaug

  官网：**https://github.com/aleju/imgaug**
  imgaug是常用的第三方数据扩增库，提供了多样的数据扩增方法，且组合起来非常方便，速度较快；

- #### albumentations

  官网：**https://albumentations.readthedocs.io**
  是常用的第三方数据扩增库，提供了多样的数据扩增方法，对图像分类、语义分割、物体检测和关键点检测都支持，速度较快。

## 四、pytorch读取路径、标签等数据

pytorch读取数据的逻辑为：先利用dataset的类封装数据，接着用dataloader对封装数据按batch读取。

SVHNdataset的类定义如下：

```
class SVHNDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        # 原始SVHN中类别10为数字0
        lbl = np.array(self.img_label[index], dtype=np.int)
        # print(type(lbl))
        lbl = list(lbl) + (6 - len(lbl)) * [10]

        return img, torch.from_numpy(np.array(lbl[:5]))

    def __len__(self):
        return len(self.img_path)
```

定义的SVHNDataset继承自pytorch中的dataset类，父类中本来定义的函数有getitem和len，将类实例对象初始化为有路径和标签的格式，并且将所有对象的字符长度补足为6个。

接着开始读取数据：

```
train_path = glob.glob('.../input/train/*.png')#以列表形式返回图片路径
train_path.sort()#将路径按升序排列
#print(train_path)
train_json = json.load(open('.../input/train.json'))#以字典形式返回json文件
train_label = [train_json[x]['label'] for x in train_json]#读取json文件中的label数据，列表形式
#print(train_label)
print(len(train_path), len(train_label))
```

将读取的路径和标签传入类实例对象：

```
data = SVHNDataset(train_path, train_label,
          transforms.Compose([
              # 缩放到固定尺寸
              transforms.Resize((64, 128)),

              # 随机颜色变换
              transforms.ColorJitter(0.2, 0.2, 0.2),

              # 加入随机旋转
              transforms.RandomRotation(5),

              # 将图片转换为pytorch 的tesntor
              # transforms.ToTensor(),

              # 对图像像素进行归一化
              # transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ]))
```

至此，用封装好的类读取图像的的工作已经完成，可到此还没有结束，我们还需要利用dataloader对dataset进行封装并且批量读取。

dataset和dataloader的作用分别如下：

- Dataset：对数据集的封装，提供索引方式的对数据样本进行读取
- DataLoder：对Dataset进行封装，提供批量读取的迭代读取

加入dataloader后，将原先data之后的代码修改如下：

```
train_loader = torch.utils.data.DataLoader(
    SVHNDataset(train_path, train_label,
                transforms.Compose([
                    transforms.Resize((64, 128)),
                    transforms.RandomCrop((60, 120)),
                    transforms.ColorJitter(0.3, 0.3, 0.2),
                    transforms.RandomRotation(5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])), 
    batch_size=40, 
    shuffle=True, 
    num_workers=0,
)
```

在加入DataLoder后，数据按照批次获取，每批次调用Dataset读取单个样本进行拼接。此时data的格式为：

torch.Size([10, 3, 64, 128]), torch.Size([10, 6])

前者为图像文件，为batchsize * chanel * height * width次序；后者为字符标签。

## 五、小结

本章对数据读取进行了详细的讲解，并介绍了常见的数据扩增方法和使用，最后使用Pytorch框架对本次赛题的数据进行读取。

from torchvision.datasets import FashionMNIST #从这个模块导入FashionMNIST数据集
from torchvision import transforms #导入数据转换处理的工具
import numpy as np  #导入数据处理的工具
import torch.utils.data as Data #与dataset和dataloader有关
import matplotlib.pyplot as plt

train_data = FashionMNIST(root='./data',
                          train=True,
                          transform=transforms.Compose([transforms.Resize(size=224), transforms.ToTensor()]),
                          download=True) # 创建FashionMNIST的训练数据集对象。指定了数据集的存储路径 (root='./data')，下载训练集 (train=True, False下载测试集)，使用transforms.Compose将多个数据转换操作组合起来。transforms.Resize(size=224)将图像的大小调整为224x224像素。transforms.ToTensor()将图像转换为PyTorch张量。download=True: 如果数据集未在指定的root目录中找到，将下载FashionMNIST数据集。
train_loader = Data.DataLoader(dataset=train_data,
                               batch_size=64,
                               shuffle=True,
                               num_workers=0) # 创建一个用于训练的数据加载器。这个数据加载器会从train_data中加载数据，每个批次大小为64 (batch_size=64)，并在每个epoch之前对数据进行随机洗牌 (shuffle=True)。num_workers=0表示不使用多进程加载数据。

# 获得一个Batch的数据
for step, (b_x, b_y) in enumerate(train_loader): #将 train_loader 转换为一个可迭代对象，返回的每个元素都是一个包含两个元素的元组 (step, (b_x, b_y))，其中 step 是当前批次的索引，(b_x, b_y) 是批次中的输入数据和对应的标签。
    if step > 0:
        break #if step > 0: break 是为了在获取到第一个批次后就跳出循环，因为这段代码可能是为了演示目的，不需要遍历整个数据集。
batch_x = b_x.squeeze().numpy()  # 移除四维张量中尺寸为1的维度，并转换成Numpy数组
batch_y = b_y.numpy()  # 将张量转换成Numpy数组
class_label = train_data.classes  # 训练集的标签
print(class_label)
print("The size of batch in train data:", batch_x.shape)  # 每个mini-batch的维度是64*224*224

# 可视化一个Batch的图像
plt.figure(figsize=(12, 5)) #创建图形窗口
for ii in np.arange(len(batch_y)):  # 使用 NumPy 的 arange 函数创建一个包含从 0 到 len(batch_y)-1 的整数的 NumPy 数组。循环遍历批次中的每个样本
    plt.subplot(4, 16, ii + 1) #设置子图，4行16列
    plt.imshow(batch_x[ii, :, :], cmap=plt.cm.gray) #使用 imshow 函数显示图像 第几个 长 宽
    plt.title(class_label[batch_y[ii]], size=10) #给每个子图添加标题，标题是图像对应的类别标签
    plt.axis("off") #设置关闭坐标轴，因为这里只关注图像的可视化而不需要坐标轴。
    plt.subplots_adjust(wspace=0.05) #调整子图之间间距的函数。具体来说，wspace 参数表示水平方向上的子图之间的间距
plt.show()

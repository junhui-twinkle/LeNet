import torch
from torch import nn #包含神经网络的一些层
from torchsummary import summary #展示模型参数

class LeNet(nn.Module):  #定义了一个名为 LeNet 的类，并指定它继承自 nn.Module
    def __init__(self): #实例化通常通过调用类的构造函数（__init__ 方法）来完成，初始化对象
        super(LeNet, self).__init__()  #在子类中调用父类构造函数
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2)
        self.sig = nn.Sigmoid()
        self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.s4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()
        self.f5 = nn.Linear(400,120)
        self.f6 = nn.Linear(120,84)
        self.f7 = nn.Linear(84,10)
    def forward(self, x):
        x = self.sig(self.c1(x))
        x = self.s2(x)
        x = self.sig(self.c3(x))
        x = self.s4(x)
        x = self.flatten(x)
        x = self.f5(x)
        x = self.f6(x)
        x = self.f7(x)
        return x

if __name__ =="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model, input_size=(1, 28, 28)))

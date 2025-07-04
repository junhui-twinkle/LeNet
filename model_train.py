import copy
import time

import torch
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from model import LeNet
import torch.nn as nn
import pandas as pd


def train_val_data_process(): #定义一个训练集和验证集处理的函数
    train_data = FashionMNIST(root='./data',
                              train=True,
                              transform=transforms.Compose([transforms.Resize(size=28), transforms.ToTensor()]),
                              download=True) #创建训练集对象
    #Data.random_split 将训练数据集划分为训练集和验证集。接受两个参数，第一个参数是要分割的数据集，第二个参数是一个整数列表，表示每个子集应包含的样本数量。
    train_data, val_data = Data.random_split(train_data, [round(0.8*len(train_data)),
                                                          round(0.2*len(train_data))])
    #创建数据加载器
    train_dataloader = Data.DataLoader(dataset=train_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=2)

    val_dataloader = Data.DataLoader(dataset=val_data,
                                       batch_size=64,
                                       shuffle=True,
                                       num_workers=2)

    return train_dataloader, val_dataloader

def train_model_process(model, train_dataloader, val_dataloader, num_epochs): #定义模型训练函数，传入model，训练集，验证集，轮次
    # 设定训练所用到的设备，有GPU用GPU没有GPU用CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 定义优化器 使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # 定义损失函数 损失函数为交叉熵函数
    criterion = nn.CrossEntropyLoss() #回归一般用均方误差函数 分类用交叉熵函数
    # 将模型放入到训练设备中
    model = model.to(device)
    # 复制当前模型的参数
    best_model_wts = copy.deepcopy(model.state_dict()) #model.state_dict()返回包含模型所有参数（权重和偏置等）的字典。copy.deepcopy()深度拷贝是为了确保 best_model_wts 不与原模型的参数共享内存，而是拥有一个独立的副本。
    #深度拷贝是为了防止在后续的训练中对 model.state_dict() 的修改也影响到 best_model_wts。

    # 初始化参数
    # 最高准确度
    best_acc = 0.0
    # 训练集损失列表
    train_loss_all = []
    # 验证集损失列表
    val_loss_all = []
    # 训练集准确度列表
    train_acc_all = []
    # 验证集准确度列表
    val_acc_all = []
    # 当前时间
    since = time.time()

    for epoch in range(num_epochs): #定义循环过程训练模型
        print("Epoch {}/{}".format(epoch, num_epochs-1)) #打印轮次
        print("-"*10) #打印类似虚线的东西

        # 初始化参数
        # 训练集损失函数
        train_loss = 0.0
        # 训练集准确度
        train_corrects = 0
        # 验证集损失函数
        val_loss = 0.0
        # 验证集准确度
        val_corrects = 0
        # 训练集样本数量
        train_num = 0
        # 验证集样本数量
        val_num = 0

        # 对每一个mini-batch训练和计算
        for step, (b_x, b_y) in enumerate(train_dataloader): #遍历训练集，取输入的特征与标签值 b_x =128*28*28*1 b_y = 128label
            # 将特征放入到训练设备中
            b_x = b_x.to(device)
            # 将标签放入到训练设备中
            b_y = b_y.to(device)
            # 设置模型为训练模式
            model.train() # PyTorch 中用于将模型设置为训练模式的方法

            # 前向传播过程，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1)  #softmax得到最大概率下标的值 沿着维度1取最大值的索引
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y) #（模型输出与b_y的值）

            # 将梯度初始化为0
            optimizer.zero_grad() #避免梯度累加
            # 反向传播计算
            loss.backward() #用损失函数反向传播计算梯度与梯度存储
            # 根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值的作用
            optimizer.step() #更新优化参数
            # 对损失函数进行累加
            train_loss += loss.item() * b_x.size(0) #loss.item()得到的是每个样本loss的平均值
            # 如果预测正确，则准确度train_corrects加1
            train_corrects += torch.sum(pre_lab == b_y.data)
            # 获取整个训练的样本数量 b_x.size(0)获取一个batch有多少样本
            train_num += b_x.size(0)

        for step, (b_x, b_y) in enumerate(val_dataloader): #验证过程的撰写，不参与反向传播
            # 将特征放入到验证设备中
            b_x = b_x.to(device)
            # 将标签放入到验证设备中
            b_y = b_y.to(device)
            # 设置模型为评估模式
            model.eval()
            # 前向传播，输入为一个batch，输出为一个batch中对应的预测
            output = model(b_x)
            # 查找每一行中最大值对应的行标
            pre_lab = torch.argmax(output, dim=1) #torch.argmax(output, dim=1) 操作是在每个样本的预测概率（或分数）中找到最大值的索引，而且是在整个 batch 上同时进行的。
            # 计算每一个batch的损失函数
            loss = criterion(output, b_y)


            # 对损失函数进行累加
            val_loss += loss.item() * b_x.size(0)
            # 如果预测正确，则准确度val_corrects加1
            val_corrects += torch.sum(pre_lab == b_y.data)
            # 当前用于验证的样本数量
            val_num += b_x.size(0) #0是指第一个维度

        # 计算并保存每一次迭代的loss值和准确率 1轮50000张数据，一个批次128
        # 计算并保存一轮训练集的loss值
        train_loss_all.append(train_loss / train_num) #将计算得到的平均训练损失添加到列表 train_loss_all 中
        # 计算并保存训练集的准确率
        train_acc_all.append(train_corrects.double().item() / train_num) #将计算得到的平均准确度添加到列表，item() 将结果转换为 Python 标量（普通的数值类型）。

        # 计算并保存一轮验证集的loss值
        val_loss_all.append(val_loss / val_num)
        # 计算并保存验证集的准确率
        val_acc_all.append(val_corrects.double().item() / val_num)

        #打印损失与准确度 .4f表示保留小数点后四位 {}表示要填值 .format (格式化) [-1]表示最后一个值
        print("{} train loss:{:.4f} train acc: {:.4f}".format(epoch, train_loss_all[-1], train_acc_all[-1]))
        print("{} val loss:{:.4f} val acc: {:.4f}".format(epoch, val_loss_all[-1], val_acc_all[-1]))

        if val_acc_all[-1] > best_acc:
            # 保存当前最高准确度
            best_acc = val_acc_all[-1]
            # 保存当前最高准确度的模型参数
            best_model_wts = copy.deepcopy(model.state_dict())

        # 计算训练和验证的耗时时间
        time_use = time.time() - since
        print("训练和验证耗费的时间{:.0f}m{:.0f}s".format(time_use//60, time_use%60))

    # 选择最优参数，保存最优参数的模型
    torch.save(best_model_wts, "F:/code/Pao/LeNet/best_model.pth")

    train_process = pd.DataFrame(data={"epoch":range(num_epochs),
                                       "train_loss_all":train_loss_all,
                                       "val_loss_all":val_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_acc_all":val_acc_all})
    #轮次 损失值 准确度
    return train_process


def matplot_acc_loss(train_process):
    # 显示每一次迭代后的训练集和验证集的损失函数和准确率
    plt.figure(figsize=(12, 4)) #图的大小
    plt.subplot(1, 2, 1) #总行总列 索引
    plt.plot(train_process['epoch'], train_process.train_loss_all, "ro-", label="Train loss") #x轴 y轴红色圆形实线图例
    plt.plot(train_process['epoch'], train_process.val_loss_all, "bs-", label="Val loss") #蓝色方形实线
    plt.legend() #显示图例的函数
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.subplot(1, 2, 2)
    plt.plot(train_process['epoch'], train_process.train_acc_all, "ro-", label="Train acc")
    plt.plot(train_process['epoch'], train_process.val_acc_all, "bs-", label="Val acc")
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # 加载需要的模型
    LeNet = LeNet()
    # 加载数据集
    train_data, val_data = train_val_data_process()
    # 利用现有的模型进行模型的训练
    train_process = train_model_process(LeNet, train_data, val_data, num_epochs=20)
    matplot_acc_loss(train_process)

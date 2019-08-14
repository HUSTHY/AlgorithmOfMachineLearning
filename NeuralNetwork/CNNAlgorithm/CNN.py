#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   CNN.py
@Time    :   2019/8/14 10:26
@Desc    :

'''
import torch
import torch.nn as nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
import torchvision
torch.manual_seed(1)    # reproducible

# Hyper Parameters
EPOCH = 1           # 训练整批数据多少次, 为了节约时间, 我们只训练一次
BATCH_SIZE = 50
LR = 0.001          # 学习率
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 False,True表示你没有下载好数据

train_data=torchvision.datasets.MNIST(#从网上下载数据集并做转换
    root='./mnist/',
    train=True,#表示的就是训练数据
    transform=torchvision.transforms.ToTensor(),#转换成 PIL.Image or numpy.ndarray
    download=DOWNLOAD_MNIST
)
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)
# print(train_loader)
test_data = torchvision.datasets.MNIST(root='./mnist/', train=False)
# print(len(test_data.test_data))
test_x=torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:1000]/255. # shape from (10000, 28, 28) to (10000, 1, 28, 28), value in range(0,1)
test_y=test_data.targets[:1000]

class CNN(nn.Module):
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1=nn.Sequential(# input shape (1, 28, 28)
            nn.Conv2d(
                    in_channels=1,# input height
                    out_channels=16 ,# n_filters
                    kernel_size=5, # filter size
                    stride=1,
                    padding=2#如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1

            ), # output shape (16, 28, 28)
            nn.ReLU(),#激活函数
            nn.MaxPool2d(kernel_size=2) # 在 2x2 空间里向下采样, output shape (16, 14, 14)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 14, 14)
            nn.Conv2d(
                in_channels=16,  # input height
                out_channels=32, # n_filters
                kernel_size=5,  # filter size
                stride=1,
                padding=2# 如果想要 con2d 出来的图片长宽没有变化, padding=(kernel_size-1)/2 当 stride=1

            ),  # output shape (32, 14, 14)
            nn.ReLU(),  # 激活函数
            nn.MaxPool2d(kernel_size=2)  # 在 2x2 空间里向下采样, output shape (32, 7, 7)
        )
        self.out=nn.Linear(32*7*7,10)

    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1) # 展平多维的卷积图成 (batch_size, 32 * 7 * 7)
        output=self.out(x)
        return output



def showData(train_data,num):
    import random
    tr_data = train_data.train_data
    label=train_data.train_labels
    indexs=[]
    for i in range(num):
        indexs.append(random.randint(0,len(tr_data)))
    for index in indexs:
        plt.imshow(tr_data[index].numpy(),cmap='gray')
        plt.title('%i'%(label[index]))
        plt.show()
def trainCNN(cnn):
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

    for epoch in range(EPOCH):
        for step, (b_x,b_y) in enumerate(train_loader):
            output=cnn(b_x)
            loss=loss_func(output,b_y)#交叉信息熵
            optimizer.zero_grad()#把梯度清0
            loss.backward() #后向计算梯度
            optimizer.step() #应用梯度
            if step%50==0:
                test_output=cnn(test_x)
                pred_y=torch.max(test_output,1)[1].data.numpy().squeeze()#把tensor转为numpy()类型
                accuracy = sum(pred_y == test_y.numpy()) / test_y.size(0)
                print('EPOCH: %d | step: %d | train loss : %.4f  | test accuray: %.4f'%(epoch,step,loss,accuracy))

    test_output = cnn(test_x[:20])
    pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
    print(pred_y, 'prediction number')
    print(test_y[:20].numpy(), 'real number')
    print(cnn)


if __name__ == '__main__':
    # showData(train_data,5)
    cnn=CNN()
    trainCNN(cnn)

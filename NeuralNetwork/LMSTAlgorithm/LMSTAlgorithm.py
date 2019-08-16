#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   LMSTAlgorithm.py
@Time    :   2019/8/16 9:14
@Desc    :

'''

'''
LMST算法是有时间属性的，
图片识别中，把每一行的数据作为一个时间点的识别数据
有N行就有N个状态


'''









import torch
import torch.nn  as nn
import torchvision
import torch.utils.data as Data
torch.manual_seed(1)    # reproducible

EPOCH=1
BATCH_SIZE=64
TIME_STEP=28
INPUT_SIZE=28
lR=0.01
DOWNLOAD_MNIST = False  # 如果你已经下载好了mnist数据就写上 False,True表示你没有下载好数据

train_data=torchvision.datasets.MNIST(
    root='./mnist',
    transform=torchvision.transforms.ToTensor(),
    train=True,
    download=DOWNLOAD_MNIST
)
# 批训练 50samples, 1 channel, 28x28 (50, 1, 28, 28)
train_loader=Data.DataLoader(dataset=train_data,batch_size=BATCH_SIZE,shuffle=True)

test_data=torchvision.datasets.MNIST(
    root='./mnist/',
    transform=torchvision.transforms.ToTensor(),
    train=False,
    download=DOWNLOAD_MNIST
)

test_x=torch.unsqueeze(test_data.data,dim=1).type(torch.FloatTensor)[:1000]/255
test_y=test_data.targets.numpy()[:1000]
# print(len(test_y))


class RNN(nn.Module):
    def __init__(self):
        super(RNN,self).__init__()

        self.rnn=nn.LSTM(
            input_size=INPUT_SIZE,
            hidden_size=64, #隐藏层的尺寸
            num_layers=1,
            batch_first=True #input里面的维度 （batch ,time_step,input_szie)
        )
        self.out = nn.Linear(64,10)

    def forward(self,x):
        r_out,(h_n,h_c)=self.rnn(x,None) #（batch ,time_step,input_szie) (h_n,h_c) 隐藏的状态：一个是主线的，一个是支线的可以这么理解
        out=self.out(r_out[:,-1,:]) #（batch ,time_step,input_szie)
        return out

rnn=RNN()
optimizer=torch.optim.Adam(rnn.parameters(),lr=lR) #优化所有参数，
loss_func=nn.CrossEntropyLoss()   # the target label is not one-hotted

for epoch in range(EPOCH):
    for step,(x,y) in enumerate(train_loader):
        x=x.view(-1,28,28)
        output=rnn(x)
        loss=loss_func(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step%50==0:
            test_out=rnn(test_x.view(-1,28,28))
            pre_y=torch.max(test_out,1)[1].numpy().squeeze()
            accuracy= sum(pre_y==test_y)/ (len(test_y))
            print('EPOCH: %d | step: %d | train loss : %.4f  | test accuray: %.4f' % (epoch, step, loss, accuracy))


test_output = rnn(test_x[:20].view(-1,28,28))
# print(test_x[1])
# print(test_x[1].size())
# print(test_x.size())
pred_y = torch.max(test_output, 1)[1].data.numpy().squeeze()
print(pred_y, 'prediction number')
print(test_y[:20], 'real number')


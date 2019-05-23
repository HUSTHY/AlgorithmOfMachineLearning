#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   stepLinearRegression.py
@Time    :   2019/5/22 21:36
@Desc    :   前向逐步线性回归算法

'''

import numpy as np
import matplotlib.pyplot as plt
def loadDataSet(fileName):
    with open(fileName) as file:
        con=file.readlines()
    x_cordArr=[];y_cordArr=[]
    for ele in con:
        line=ele.strip().split('\t')
        temp=[]
        for i in range(len(line)-1):
            temp.append(float(line[i]))
        x_cordArr.append(temp)
        y_cordArr.append(float(line[-1]))
    return x_cordArr,y_cordArr
def calculateError(yMat,yPredict):
    m=np.shape(yMat)[0]
    error=sum((yMat-yPredict).T*(yMat-yPredict))/m
    return error

def stepLinearRegression(xMat,yMat,step,maxIter):
    """
        Description:前向逐步回归算法：每次迭代引入每个特征的回归系数，然后经过多次迭代，
        同时在迭代的过程中计算每次预测的误差，选取误差小的那个作为引入的变量。
        Params:
                xMat,yMat,
                step——每次迭代的步长
                maxIter——最大迭代次数
        Return:

        Author:
                HY
        Modify:
                2019/5/22 22:34
    """
    m,n=np.shape(xMat)
    ws=np.zeros((n,1))#初始的回归系数为0
    returnwsMat=np.zeros((maxIter,n))
    wsB=ws.copy()
    for j in range(maxIter):
        lowestError = float('inf')
        for i in range(n):#遍历每个特征
            for sign in [-1, 1]:#回归系数给与一个步长的改变
                wsT = ws.copy()
                wsT[i] += step * sign
                yPredict = xMat * wsT#预测分类
                error = calculateError(yMat, yPredict)#计算误差
                if lowestError >= error:#得到最小的误差和相应的回归系数
                    lowestError = error
                    wsB = wsT
        ws=wsB.copy()
        returnwsMat[j,:]=ws.T
    return returnwsMat

def showWs(WsMat):
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,7))
    #画一个矩阵m*n 就会得到那条曲线；每一条曲线表示的是矩阵每一列的值随着序号增大的变化
    plt.plot(WsMat)
    plt.title('前向逐步线性回归回归系数和迭代次数的关系！')
    plt.show()

def standardlized( x, y):
    xMat=np.mat(x);yMat=np.mat(y)
    x_mean=np.mean(x)
    y_mean=np.mean(y)
    x_var=np.var(x,axis=0)
    xMat=(xMat-x_mean)/x_var
    yMat=yMat-y_mean
    return xMat,yMat.T


if __name__ == '__main__':
    x, y = loadDataSet('abalone.txt')
    xMat,yMat=standardlized( x, y)
    wsMat = stepLinearRegression(xMat,yMat,0.005,500)
    print(wsMat[400:450,:])
    showWs(wsMat)


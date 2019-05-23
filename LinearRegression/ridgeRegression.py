#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   ridgeRegression.py
@Time    :   2019/5/22 17:53
@Desc    :   岭回归算法的实现

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

def ridgeRrgression(xMat,yMat,lamda):
    """
        Description:岭回归算法——得出模型的回归系数
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/22 21:40
    """
    I=np.eye(np.shape(xMat)[1])
    xTx=xMat.T*xMat+I*lamda
    if np.linalg.det(xTx)==0:
        print('矩阵行列式为0！结束！')
        return
    ws=xTx.I*xMat.T*yMat
    return ws

def ridgeRrgressionTest(x,y):
    xMat=np.mat(x)
    yMat=np.mat(y).T
    x_mean=np.mean(xMat)
    y_mean=np.mean(yMat)
    x_var=np.var(xMat,axis=0)
    xMat=(xMat-x_mean)/x_var
    yMat=yMat-y_mean
    # 以上过程对X进行了标准化——0均值，单位方差
    num=30
    wsMat=np.zeros((num,np.shape(xMat)[1]))
    for i in range(num):
        ws=ridgeRrgression(xMat,yMat,np.exp(i-15))
        wsMat[i,:]=ws.T
    return wsMat

def showWs(WsMat):
    """
        Description:显示模型的回归系数不断随着lamda增大而缩小的过程
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/22 21:38
    """
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,6))
    #画一个矩阵m*n 就会得到那条曲线；每一条曲线表示的是矩阵每一列的值随着序号增大的变化
    plt.plot(WsMat)
    plt.title('岭回归lamda和回归系数的关系')
    plt.show()

if __name__ == '__main__':
    x,y=loadDataSet('abalone.txt')
    wsMat=ridgeRrgressionTest( x,y)
    print(wsMat)
    showWs(wsMat)

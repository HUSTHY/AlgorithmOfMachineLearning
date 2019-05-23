#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   linearRegression.py
@Time    :   2019/5/20 19:07
@Desc    :

'''

import matplotlib.pyplot as plt
import numpy as np

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

def showData(x_cordArr,y_cordArr):
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(13,20))
    x=np.zeros(len(x_cordArr))
    yMat = np.mat(y_cordArr)
    for i in range(len(x_cordArr)):
        x[i]=x_cordArr[i][1]
    sortId=x.argsort()
    axes0 = plt.subplot(511)
    axes0.scatter(x,y_cordArr,c='b',s=20,alpha=0.5)
    y_predict = np.mat(x_cordArr) * computeRegression(x_cordArr,y_cordArr)
    coef = np.corrcoef(y_predict.T, yMat)
    axes0.plot(x[sortId],y_predict[sortId],c='r',lw=2,label='相关系数：%0.4f'%coef[0][1])
    axes0.legend()


    axes1 = plt.subplot(512)
    y_predict=lwlrComputeResult(x_cordArr,y_cordArr,1)
    coef = np.corrcoef(y_predict.T, yMat)
    axes1.scatter(x, y_cordArr, c='b', s=20, alpha=0.5)
    axes1.plot(x[sortId],y_predict[sortId],c='r',lw=2,label='相关系数：%0.4f'%coef[0][1])
    axes1.legend()

    axes2 = plt.subplot(513)
    y_predict = lwlrComputeResult(x_cordArr, y_cordArr, 0.1)
    coef = np.corrcoef(y_predict.T, yMat)
    axes2.scatter(x, y_cordArr, c='b', s=20, alpha=0.5)
    axes2.plot(x[sortId],y_predict[sortId],c='r',lw=2,label='相关系数：%0.4f'%coef[0][1])
    axes2.legend()

    axes3 = plt.subplot(514)
    y_predict = lwlrComputeResult(x_cordArr, y_cordArr, 0.03)
    coef = np.corrcoef(y_predict.T, yMat)
    axes3.scatter(x, y_cordArr, c='b', s=20, alpha=0.5)
    axes3.plot(x[sortId],y_predict[sortId],c='r',lw=2,label='相关系数：%0.4f'%coef[0][1])
    axes3.legend()

    axes4 = plt.subplot(515)
    y_predict = lwlrComputeResult(x_cordArr, y_cordArr, 0.004)
    coef = np.corrcoef(y_predict.T, yMat)
    axes4.scatter(x, y_cordArr, c='b', s=20, alpha=0.5)
    axes4.plot(x[sortId],y_predict[sortId],c='r',lw=2,label='相关系数：%0.4f'%coef[0][1])
    axes4.legend()

    axes0_title = axes0.set_title('普通线性回归')
    axes1_title = axes1.set_title('局部加权线性回归，k=1')
    axes2_title = axes2.set_title('局部加权线性回归，k=0.1')
    axes3_title = axes3.set_title('局部加权线性回归，k=0.01')
    axes4_title = axes4.set_title('局部加权线性回归，k=0.001')

    plt.setp(axes0_title)
    plt.setp(axes1_title)
    plt.setp(axes2_title)
    plt.setp(axes3_title)
    plt.setp(axes4_title)



    plt.xlabel('X轴')
    plt.show()

def computeRegression(x_cordArr,y_cordArr):
    """
        Description:显性回归的回归系数求法
        Params:
                W——回归系数
        Return:

        Author:
                HY
        Modify:
                2019/5/21 15:15
    """
    xMat=np.mat(x_cordArr);yMat=np.mat(y_cordArr).T
    XTX=xMat.T*xMat
    if np.linalg.det(XTX)==0:
        print('XTX的行列式为0，其逆矩阵不存在')
        return
    #直接根据公式求得回归系数
    W=XTX.I*xMat.T*yMat
    return W


def lwlrComputeWeights(testPoint,x,y,k):
    """
        Description:
        Params:
                testPoint——要测试的点
                x——样本所有点
                y——标签
                k——核系数
        Return:

        Author:
                HY
        Modify:
                2019/5/21 18:36
    """
    m=len(y)
    xMat=np.mat(x)
    yMat=np.mat(y).T
    weihts=np.eye(m)
    for i in range(m):
        diffMat=testPoint-xMat[i,:]
        weihts[i,i]=np.exp(diffMat*diffMat.T/(-2*k*k))
    if np.linalg.det(xMat.T*weihts*xMat)==0:
        print('矩阵XWX是不可逆矩阵！')
        return
    # 直接根据公式求得回归系数
    ws=(xMat.T*weihts*xMat).I*xMat.T*weihts*yMat
    return testPoint*ws

def lwlrComputeResult(x_cordArr, y_cordArr,k):
    m=len(x_cordArr)
    yPredictMat=np.zeros(m)
    for i in range(m):
        yPredictMat[i]=lwlrComputeWeights(x_cordArr[i],x_cordArr,y_cordArr,k)
    return  yPredictMat


if __name__ == '__main__':
    x_cordArr, y_cordArr=loadDataSet('testDataSet.txt')
    showData(x_cordArr, y_cordArr)

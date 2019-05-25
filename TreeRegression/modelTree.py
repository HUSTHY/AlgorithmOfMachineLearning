#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   modelTree.py
@Time    :   2019/5/25 19:18
@Desc    :   模型树的实现
            它和CART算法不同的是，CART叶节点上是单个数值——平均值；而modelTree则是线性回归的回归系数ws

'''

import numpy as np
import matplotlib.pyplot as plt
def linearSolve(dataSet):
    """
        Description:线性回归的拟合——回归系数的求取
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/25 19:50
    """
    m,n=np.shape(dataSet)
    X=np.mat(np.ones((m,n)));Y=np.mat(np.ones((m,1)))
    X[:,1:n]=dataSet[:,0:n-1];Y=dataSet[:,-1]
    xTx=X.T*X
    if np.linalg.det(xTx)==0:
        print('逆矩阵不存在！')
    ws=xTx.I*X.T*Y
    return ws,X,Y


def  modelLeaf(dataSet):
    """
        Description:叶节点就是一个回归系数
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/25 19:51
    """
    ws, X, Y=linearSolve(dataSet)
    return ws

def modelError(dataSet):
    """
        Description:误差的计算
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/25 19:51
    """
    ws, X, Y = linearSolve(dataSet)
    Yhat=X*ws
    return (Y-Yhat).T*(Y-Yhat)


def loadDataSet(fileName):
    with open(fileName) as file:
        con = file.readlines()
    dataSet=[]
    for ele in con:
        line=ele.strip().split('\t')
        line=[float(l) for l in line]
        dataSet.append(line)
    return dataSet
def binSplitDataTest(dataSet,featureIdex,featureVal):
    """
        Description:树回归算法中，根据特征二元切分数据集
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/25 16:13
    """
    #注意np.nonzero的作用
    rightMat=dataSet[np.nonzero(dataSet[:,featureIdex]>featureVal)[0]]
    leftMat=dataSet[np.nonzero(dataSet[:,featureIdex]<=featureVal)[0]]
    return  rightMat,leftMat


def chooseBestSplit(dataSet,leafType=modelLeaf,errorType=modelError,ops=(1,4)):
    """
        Description:遍历所有的特征和特征的值，选择出最佳的切分点
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/25 16:14
    """
    tolS=ops[0];tolN=ops[1]#停止条件，一个是误差，一个是分割数据的最小范围
    if len(set(dataSet[:,-1].T.tolist()[0]))==1:#停止条件，数据集中元素相同停止
        return None,leafType(dataSet) #返回叶子的值
    S=errorType(dataSet)
    bestFeatureIndex=0;bestFeatureVal=0
    m,n =np.shape(dataSet)
    smallestS=float('inf')
    for feaIn in range(n-1):#遍历所有的特征
        for splitVal in set(dataSet[:,feaIn].T.A.tolist()[0]):#根绝特征值来遍历
            rightMat, leftMat=binSplitDataTest(dataSet,feaIn,splitVal)#得到切分后的数据集
            if np.shape(rightMat)[0]<tolN or np.shape(leftMat)[0]<tolN:
                continue
            newS=errorType(rightMat)+errorType(leftMat)#计算误差——均方差
            if newS<smallestS:#得到最大值
                smallestS=newS
                bestFeatureIndex=feaIn
                bestFeatureVal=splitVal
    if (S-smallestS)<tolS:
        return None,leafType(dataSet)
    rightMat, leftMat = binSplitDataTest(dataSet, feaIn, splitVal)
    if np.shape(rightMat)[0] < tolN or np.shape(leftMat)[0]< tolN:
        return None,leafType(dataSet)
    return bestFeatureIndex,bestFeatureVal

def createModelTree(dataSet,leafType,errorType,ops):
    """
        Description:递归生成回归树
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/25 16:18
    """
    #递归地生成回归树
    RegressionTree={}#树的定义——也就是一个字典
    bestFeatureIndex, bestFeatureVal=chooseBestSplit(dataSet,leafType,errorType,ops)
    if bestFeatureIndex==None:
        return bestFeatureVal
    RegressionTree['bestFeatureIndex']=bestFeatureIndex
    RegressionTree['spVal'] = bestFeatureVal
    rightMat, leftMat = binSplitDataTest(dataSet, bestFeatureIndex, bestFeatureVal)
    RegressionTree['rightMat'] = createModelTree(rightMat,leafType,errorType,ops)
    RegressionTree['leftMat'] = createModelTree(leftMat,leafType,errorType,ops)
    return RegressionTree

def plotDataSet(fileName):
    dataMat = loadDataSet(fileName)  # 加载数据集
    n = len(dataMat)  # 数据个数
    xcord = [];
    ycord = []  # 样本点
    for i in range(n):
        xcord.append(dataMat[i][0]);
        ycord.append(dataMat[i][1])  # 样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()
if __name__ == '__main__':
    dataSet = np.mat(loadDataSet('exp2.txt'))
    modelTree = createModelTree(dataSet, modelLeaf, modelError, (1, 4))
    print(modelTree)
    plotDataSet('exp2.txt')
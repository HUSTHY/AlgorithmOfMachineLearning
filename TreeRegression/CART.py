#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   CART.py
@Time    :   2019/5/24 16:36
@Desc    :   CART算法的实现

'''

import numpy as np
import matplotlib.pyplot as plt

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

def regLef(dataSet):
    #把数据集矩阵展开成一行
    return np.mean(dataSet[:,-1])

def resErr(dataSet):
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]


def chooseBestSplit(dataSet,leafType=regLef,errorType=resErr,ops=(1,4)):
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
        return None,leafType(dataSet)
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

def createRegressionTree(dataSet,leafType,errorType,ops):
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
    RegressionTree['rightMat'] = createRegressionTree(rightMat,leafType,errorType,ops)
    RegressionTree['leftMat'] = createRegressionTree(leftMat,leafType,errorType,ops)
    return RegressionTree

def plotDataSet0(filename):
    dataMat = loadDataSet(filename)  # 加载数据集
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

def plotDataSet1(filename):
    dataMat = loadDataSet(filename)  # 加载数据集
    n = len(dataMat)  # 数据个数
    xcord = [];
    ycord = []  # 样本点
    for i in range(n):
        xcord.append(dataMat[i][1]);
        ycord.append(dataMat[i][2])  # 样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.show()


def isTree(obj):
    """
        Description: 判定输入的对象是不是树
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/25 16:32
    """
    import types
    return (type(obj).__name__ == 'dict')

def getMean(tree):
    """
        Description:对树进行递归的处理得到平均值——对树进行塌陷处理(即返回树平均值)
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/25 16:32
    """
    if isTree(tree['rightMat']): tree['rightMat'] = getMean(tree['rightMat'])
    if isTree(tree['leftMat']): tree['leftMat'] = getMean(tree['leftMat'])
    return (tree['leftMat'] + tree['rightMat']) / 2.0

def prune(tree, testData):
    """
        Description:后剪枝算法
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/25 16:33
    """
    # 如果测试集为空,则对树进行塌陷处理
    if np.shape(testData)[0] == 0: return getMean(tree)
    # 如果有左子树或者右子树,则切分数据集
    if (isTree(tree['rightMat']) or isTree(tree['leftMat'])):
        lSet, rSet = binSplitDataTest(testData, tree['bestFeatureIndex'], tree['spVal'])
    # 处理左子树(剪枝)
    if isTree(tree['leftMat']): tree['leftMat'] = prune(tree['leftMat'], lSet)
    # 处理右子树(剪枝)
    if isTree(tree['rightMat']): tree['rightMat'] = prune(tree['rightMat'], rSet)
    # 如果当前结点的左右结点为叶结点
    if not isTree(tree['leftMat']) and not isTree(tree['rightMat']):
        lSet, rSet = binSplitDataTest(testData, tree['bestFeatureIndex'], tree['spVal'])
        # 计算没有合并的误差
        errorNoMerge = np.sum(np.power(lSet[:, -1] - tree['leftMat'], 2)) + np.sum(
            np.power(rSet[:, -1] - tree['rightMat'], 2))
        # 计算合并的均值
        treeMean = (tree['leftMat'] + tree['rightMat']) / 2.0
        # 计算合并的误差
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        # 如果合并的误差小于没有合并的误差,则合并
        if errorMerge < errorNoMerge:
            # print("merging")
            return treeMean
        else:
            return tree
    else:
        return tree


if __name__ == '__main__':
    # plotDataSet0('testDataSet0.txt')
    # plotDataSet1('testDataSet1.txt')
    # dataSet=np.mat(loadDataSet('testDataSet0.txt'))
    # RegressionTree=createRegressionTree(dataSet,regLef,resErr,(1,4))
    # print(RegressionTree)
    # dataSet = np.mat(loadDataSet('testDataSet1.txt'))
    # RegressionTree = createRegressionTree(dataSet, regLef, resErr, (1, 4))
    # print(RegressionTree)
    # dataSet = np.mat(loadDataSet('testDataSet2.txt'))
    # RegressionTree = createRegressionTree(dataSet, regLef, resErr, (1, 4))
    # print(RegressionTree)

    print('剪枝前:')
    train_filename = 'trainDataSet3.txt'
    train_Data = np.mat(loadDataSet(train_filename))
    train_Mat = np.mat(train_Data)
    tree = createRegressionTree(train_Mat,regLef, resErr, (1, 4))
    print(tree)
    print('\n剪枝后:')
    test_filename = 'testDataSet3.txt'
    test_Data = loadDataSet(test_filename)
    test_Mat = np.mat(test_Data)
    print(prune(tree, test_Mat))
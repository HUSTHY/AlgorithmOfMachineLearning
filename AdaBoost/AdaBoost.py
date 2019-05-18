#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   AdaBoost.py
@Time    :   2019/5/16 17:55
@Desc    :

'''
import  numpy as np
import matplotlib.pyplot as plt
def loadDataSet():
    dataSet=np.array([
            [1, 5],
            [2, 2],
            [3, 1],
            [4, 6],
            [6, 8],
            [6, 5],
            [7, 9],
            [8, 7],
            [9, 8],
            [10, 2]
        ])
    label=[1,1,-1,-1,1,-1,1,1,-1,-1]
    return dataSet,label

def showData(dataSet,label):
    dataSet_plus=[]
    dataSet_minus=[]
    for i in range(len(label)):
        if label[i]>0:
            dataSet_plus.append(dataSet[i])
        else:
            dataSet_minus.append(dataSet[i])
    dataSet_plus=np.array(dataSet_plus)
    dataSet_minus=np.array(dataSet_minus)
    plt.scatter(dataSet_plus[:,0],dataSet_plus[:,1],c='red',s=50,alpha=0.8,marker='+')
    plt.scatter(dataSet_minus[:,0],dataSet_minus[:,1],c='blue',s=50,alpha=0.8)
    # if Bdim==0:
    #     y = np.arange(0, 11, 2)
    #     x = 0 * y + Bthreshval
    #     plt.plot(x, y, linewidth=2, c='black')
    # else:
    #     x = np.arange(0, 11, 2)
    #     y = 0 * x + Bthreshval
    #     plt.plot(x, y, linewidth=2, c='black')

    plt.show()

def singleDesTreeClassification(dataSet,dimen,threshval,flag):
    """
        Description:单层决策树
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/17 17:29
    """
    dataSetMat=np.mat(dataSet)
    m=np.shape(dataSetMat)[0]
    resuArray=np.ones((m,1))
    if flag=='lt':
        resuArray[dataSetMat[:, dimen] <= threshval] = -1
    else:
        resuArray[dataSetMat[:, dimen] > threshval] = -1

    return resuArray

def getBestSingleDesTree(dataSet,label,D):
    """
        Description:得到误差最小的决策树，D是初始权重系数
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/17 17:30
    """
    dataSetMat=np.mat(dataSet)
    labelMat=np.mat(label).transpose()
    m,n=np.shape(dataSetMat)
    minError=float('inf')
    Bdim=0
    Bthreshval=0
    bestRes=[]
    bestTree={}
    for i in range(n):
        vmax=dataSetMat[:,i].max()
        vmin=dataSetMat[:,i].min()
        stepsize = (vmax - vmin) / 20
        for j in range(-1,21):
            for ele in ['lt','gt']:
                threshval = j*stepsize+vmin
                restlut = singleDesTreeClassification(dataSet, i, threshval,ele)
                #计算误差：和权重有关
                error = np.ones((m, 1))
                error[restlut == labelMat] = 0
                weights = np.dot(D.T, error)
                if minError > weights:
                    bestTree['minError'] = weights
                    bestTree['dim'] = i
                    bestTree['restlut'] = restlut.copy()
                    bestTree['threshval'] = threshval
                    bestTree['flag']=ele
                    Bthreshval = threshval
                    Bdim = i
                    minError = weights
                    bestRes = restlut.copy()
    return minError,bestRes,bestTree


def adaBoostTrains(dataSet, label):
    """
        Description:adaBoost主体流程的实现
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/17 17:30
    """
    dataSetMat=np.mat(dataSet)
    m=np.shape(dataSetMat)[0]
    labelMat=np.mat(label).T
    D=np.ones((m,1))/m
    maxIter=40
    alphas=[]
    results=[]
    aggClassEst=np.zeros((m,1))
    weakClassifiers=[]
    for i in range(maxIter):
        # 第一步：计算误差
        minError, bestRes,bestTree=getBestSingleDesTree(dataSet, label,D)
        results.append(bestRes)
        if minError==0:
            print('minError:',minError)
            break
        # 第二步计算alpha
        alpha=float(0.5*np.log((1-minError)/minError))
        bestTree['alpha']=alpha
        weakClassifiers.append(bestTree)
        alphas.append(alpha)
        # 第三步骤更新权重
        expVal=np.exp(-1*alpha*np.multiply(labelMat,bestRes))
        D=np.multiply(D,expVal)
        D=D/sum(D)
        aggClassEst+=alpha*bestRes
        if (np.sign(aggClassEst)==labelMat).all():
            print('训练分类完全正确，跳出循环')
            break
    return alphas,results,aggClassEst,weakClassifiers

def strongClassifer(dataSet,weakClassifiers):
    m=np.shape(dataSet)[0]
    predVal=np.zeros((m,1))
    for ele in weakClassifiers:
        predictLabel = singleDesTreeClassification(dataSet,ele['dim'],ele['threshval'],ele['flag'])
        predVal+=ele['alpha']*predictLabel
    return np.sign(predVal)


if __name__ == '__main__':
    dataSet, label=loadDataSet()
    # D=np.ones((1,10))/10
    # bestTree,Bthreshval,Bdim=getBestSingleDesTree(dataSet, label,D)
    showData(dataSet, label)
    alphas,results,aggClassEst,weakClassifiers=adaBoostTrains(dataSet, label)
    testDataSet=np.array([
        [1,3],
        [2,7],
        [9,5]
    ])
    print(strongClassifer(testDataSet,weakClassifiers))

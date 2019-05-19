#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   ClassifierEvaluation.py
@Time    :   2019/5/18 17:13
@Desc    :   ROC曲线和AUC

'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import time
def loadDataSet(fileName):
    dataSet = [];
    label = []
    with open(fileName) as file:
        con = file.readlines()
    for ele in con:
        line = ele.strip().split('\t')
        lineArr = []
        for l in range(len(line) - 1):
            lineArr.append(float(line[l]))
        dataSet.append(lineArr)
        label.append(float(line[-1]))
    label = [-1.0 if ele == 0 else ele for ele in label]
    return dataSet, label

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
        stepsize = (vmax - vmin) / 10
        for j in range(-1,11):
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

def plotROCAndAUC(aggClassEst,classLabel):
    """
            Description:分类器中的分类强度（概率型）设定一个阈值，当分类强度阈值为0是，所有的预测的分类点都被视为正样本，
            当分类强度的阈值设置为1时，所有的预测的分类点被视为负样本。——有一个最大和最小的阈值。
            从（1.0,1.0）出发，多了一个TP，y方向上移动一步；多了一个FP，X方向上移动一步。
            ROC——y轴就是TPR召回率，X轴就是FPR——假正率
            Params:

            Return:

            Author:
                    HY
            Modify:
                    2019/5/18 17:45
        """
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong'] # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False
    cur=[1.0,1.0]
    ySum=0
    count = np.sum(np.array(classLabel) == 1.0)
    xStep=float(1/(len(classLabel)-count))
    yStep=float(1/count)
    aggClassEstList=aggClassEst.ravel()
    predictStrongth=aggClassEstList.argsort().tolist()
    plt.figure(figsize=(8,6))
    for ele in predictStrongth:
        #多了一个TP，y方向上移动一步
        if classLabel[ele]==1.0:
            xDel=0;yDel=yStep
        #多了一个FP，X方向上移动一步。
        else:
            xDel=xStep;yDel=0
            ySum+=cur[1]
        plt.plot([cur[0],cur[0]-xDel],[cur[1],cur[1]-yDel],c='r',linewidth='2')
        cur=[cur[0]-xDel,cur[1]-yDel]
    #计算AUC
    AUC=ySum*xStep
    print('AUC:%f '%AUC)
    plt.plot([0,0.001],[0,0.001],c='r',label='AUC is:%0.4f'% AUC)
    plt.plot([0,1],[0,1],c='b',linewidth='2',linestyle='--')
    plt.title('ROC曲线',fontsize='15')
    plt.ylabel('TPR——召回率',fontsize='10')
    plt.xlabel('FPR——假正率',fontsize='10')
    plt.legend(fontsize='10')
    plt.show()



if __name__ == '__main__':
    dataSet, label = loadDataSet('HorseData\horseColicTraining.txt')
    time1=time.time()
    alphas, results, aggClassEst, weakClassifiers = adaBoostTrains(dataSet, label)
    time2=time.time()
    print('Finshed in % 4f'%(time2-time1))
    plotROCAndAUC(aggClassEst,label)
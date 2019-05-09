#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   WholeSMO_SVM.py
@Time    :   2019/5/8 15:37
@Desc    :

'''
import numpy as np
import random
import time
import matplotlib.pyplot as plt
class optStruct:
    def __init__(self,dataMat,classLabels,C,toler):
        self.X=dataMat
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=np.shape(dataMat)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))



def loadDataSet(fileName):
    dataArray=[];labelArray=[]
    with open(fileName) as file:
        con=file.readlines()
    for ele in con:
        line=ele.strip().split('\t')
        line=[float(e) for e in line]
        dataArray.append(line[0:len(line)-1])
        labelArray.append(line[-1])
    return dataArray,labelArray

def getDiffJ(i,m):
    """
        Description:从0到m中找到一个不等于i的j
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/8 14:34
    """
    j=i
    while j==i:
        j=int(random.uniform(0,m))
    return j

def clipped(alpha,L,H):
    """
        Description:alpha根据上下界的关系进行修剪
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/8 14:35
    """
    if alpha<L:
        return L
    elif alpha>H:
        return H
    else:
        return alpha
def calcEk(oS,k):
    fxk=float(np.multiply(oS.alphas,oS.labelMat).T*(oS.X*oS.X[k,:].T)+oS.b)
    Ek=fxk-float(oS.labelMat[k])
    return Ek
def calEta(oS,i,k):
    eta=oS.X[i,:]*oS.X[i,:].T+oS.X[k,:]*oS.X[k,:].T-2.0*oS.X[i,:]*oS.X[k,:].T
    return eta
def selectJAndOutEtaEJ(i,oS,Ei):
    maxk=-1;maxDelatE=0;Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList=np.nonzero(oS.eCache[:,0])[0]
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:continue
            Ek=calcEk(oS,k)
            eta=calEta(oS,i,k)
            if eta<=0:continue
            # deltaE = (Ei - Ek) / eta
            deltaE = (Ei - Ek)
            if abs(deltaE)>abs(maxDelatE):
                maxk=k;maxDelatE=deltaE;Ej=Ek
    else:
        maxk=getDiffJ(i,oS.m)
        # eta = calEta(oS, i, maxk)
        # while eta<=0:
        #     maxk = getDiffJ(i, ota
        #         maxDelatE = (Ei - Ej)
        #     # for k in range(oS.m):
        #     #     if i==k:continueS.m)
        #     eta = calEta(oS, i, maxk)
        # print('eta:',eta)
        Ej = calcEk(oS,maxk)
        # maxDelatE=(Ei - Ej) / e
    #     Ek = calcEk(oS, k)
    #     eta=calEta(oS,i,k)
    #     if eta<=0: continue
    #     deltaE = (Ei - Ek) / eta
    #     if abs(deltaE) > abs(maxDelatE):
    #             maxk=k;maxDelatE=deltaE;Ej=Ek
    return maxk,Ej,maxDelatE

def update(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]


def innerL(i,oS):
    Ei=calcEk(oS,i)
    if((oS.labelMat[i]*Ei<-oS.tol) and (oS.alphas[i]<oS.C)) or((oS.labelMat[i]*Ei>oS.tol)and (oS.alphas[i]>0)):
        j, Ej, maxDelatE=selectJAndOutEtaEJ(i,oS,Ei)
        alphaIold = oS.alphas[i].copy()
        alphaJold = oS.alphas[j].copy()

        # 步骤2：计算alpha的上下界
        if oS.labelMat[i] == oS.labelMat[j]:
            L = max(0, alphaJold + alphaIold -oS.C)
            H = min(oS.C, alphaJold + alphaIold)
        else:
            L = max(0, alphaJold - alphaIold)
            H = min(oS.C, oS.C + alphaJold - alphaIold)
        if L == H:
            return 0

        eta=calEta(oS,i,j)
        if eta<=0:
            return 0
        #步骤4：更新alphaJ
        oS.alphas[j] = alphaJold + oS.labelMat[j]*maxDelatE/eta
        # oS.alphas[j] = alphaJold + oS.labelMat[j] * maxDelatE
        oS.alphas[j]=clipped(oS.alphas[j],L,H)

        update(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            print('变化太小')
            return 0

        oS.alphas[i] = alphaIold + oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        update(oS, i)

        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[i, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[i, :] * oS.X[j, :].T
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.X[i, :] * oS.X[j, :].T - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.X[j, :] * oS.X[j, :].T
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def wholeSMO(dataMatIn, classLabels, C, toler, maxIter):
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler)  # 初始化数据结构
    print(oS)
    iter = 0  # 初始化当前迭代次数
    entireSet = True;
    alphaPairsChanged = 0
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):  # 遍历整个数据集都alpha也没有更新或者超过最大迭代次数,则退出循环
        alphaPairsChanged = 0
        if entireSet:  # 遍历整个数据集
            for i in range(oS.m):
                alphaPairsChanged += innerL(i, oS)  # 使用优化的SMO算法
                # print("全样本遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        else:  # 遍历非边界值
            nonBoundIs = np.nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]  # 遍历不在边界0和C的alpha
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i, oS)
                # print("非边界遍历:第%d次迭代 样本:%d, alpha优化次数:%d" % (iter, i, alphaPairsChanged))
            iter += 1
        if entireSet:  # 遍历一次后改为非边界遍历
            entireSet = False
        elif (alphaPairsChanged == 0):  # 如果alpha没有更新,计算全样本遍历
            entireSet = True
        print("迭代次数: %d" % iter)
    return  oS.alphas,oS.b

def calculateWVec(alphas,dataSetArray,labelArray):
    """
        Description:根据alphas计算W向量，使用的是矩阵的运算方式
        Params:
                alphas
                dataSetArray——训练集的数据
                labelArray——分类标签集
        Return:
                wVecArray.tolist()——向量的列表形式
        Author:
                HY
        Modify:
                2019/5/8 14:50
    """
    alphasMat=np.mat(alphas)
    labelMat=np.mat(labelArray).T
    t=np.multiply(alphasMat,labelMat).T
    wVec=t*np.mat(dataSetArray)
    wVecArray=np.array(wVec).T
    return wVecArray.tolist()

def showDatasAndClassificatonResult(dataSetArray,labelArray,wVec,b,alphas):
    """
        Description:数据的可视化。把数据点已经超平面和支持向量上的点可视化出来
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/8 14:52
    """
    data_plus=[];data_minus=[]
    for i in range(len(dataSetArray)):
        if labelArray[i]>0:
            data_plus.append(dataSetArray[i])
        else:
            data_minus.append(dataSetArray[i])
    data_plus_array=np.array(data_plus).T
    data_minus_array=np.array(data_minus).T
    plt.figure(figsize=(13,7))
    plt.scatter(data_plus_array[0],data_plus_array[1],c='red',s=50,alpha=0.8)
    plt.scatter(data_minus_array[0],data_minus_array[1],c='blue',s=50,alpha=0.8)

    dataSetMat=np.mat(dataSetArray)
    x1=dataSetMat[:,0]
    #这里的Y就是数据集中的另一维特征，并不是分类标签；令那个为0就得到了坐标
    y1=-(x1*wVec[0]+b)/wVec[1]
    plt.plot(x1,y1,c='black',alpha=0.8)
    for i,alpha in enumerate(alphas):
        if alpha>0 :
            x,y=dataSetArray[i]
            plt.scatter([x],[y],s=250,c='none',alpha=0.8,linewidths=1.0,edgecolors='black')
    plt.show()


if __name__ == '__main__':
    dataArray,labelArray=loadDataSet('testSet.txt')
    time1 = time.time()
    alphas, b = wholeSMO(dataArray,labelArray,0.6,0.001,40)
    time2 = time.time()
    print('finshed in %f seconds' % (time2 - time1))
    wVec = calculateWVec(alphas, dataArray, labelArray)
    print(wVec)
    showDatasAndClassificatonResult(dataArray, labelArray, wVec, b, alphas)


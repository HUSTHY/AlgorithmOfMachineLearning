#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   handwritingClassification.py
@Time    :   2019/5/14 17:51
@Desc    :
功能主要是实现：
SVM对手写数字的识别，使用了高斯核函数
然后是使用sklearn库中的函数来识别
注意这个是多分类问题，简单起见采用ovr策略，忽视数据倾斜的影响
'''

import numpy as np
import random
from os import listdir
import time
from sklearn import svm
from sklearn.model_selection import GridSearchCV
class optStruct:
    def __init__(self,dataMat,classLabels,C,toler,KTup):
        self.X=dataMat
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=np.shape(dataMat)[0]
        self.alphas=np.mat(np.zeros((self.m,1)))
        self.b=0
        self.eCache=np.mat(np.zeros((self.m,2)))
        #给定了K核值的定义
        self.K=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],KTup)


def kernelTrans(X,A,KTup):
    """
        Description:
        Params:
                X,A——全部数据矩阵和其中的一行
                KTup——核函数参数（'rbf',0.01）
        Return:

        Author:
                HY
        Modify:
                2019/5/15 13:35
    """
    m,n=np.shape(X)
    K=np.mat(np.zeros((m,1)))
    if KTup[0]=='lin':
        K=X*A.T
    elif KTup[0]=='rbf':
        for j in range(m):
            deltaRow=X[j,:]-A
            K[j]=deltaRow* deltaRow.T
        K=np.exp(K/(-1*KTup[1]**2))
    else: raise NameError('核函数无法识别')
    return K

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

def update(oS,k):
    Ek=calcEk(oS,k)
    oS.eCache[k]=[1,Ek]


def calcEk(oS,k):
    fxk=float(np.multiply(oS.alphas,oS.labelMat).T*oS.K[:,k]+oS.b)
    Ek=fxk-float(oS.labelMat[k])
    return Ek


def calEta(oS,i,k):
    eta=oS.K[i,i]+oS.K[k,k]-2.0*oS.K[i,k]
    return eta
def selectJAndOutEtaEJ(i,oS,Ei):
    maxk=-1;maxDelatE=0;Ej=0
    oS.eCache[i]=[1,Ei]
    validEcacheList=np.nonzero(oS.eCache[:,0])[0]
    if(len(validEcacheList))>1:
        for k in validEcacheList:
            if k==i:continue
            Ek=calcEk(oS,k)
            deltaE = (Ei - Ek)
            if abs(deltaE)>abs(maxDelatE):
                maxk=k;maxDelatE=deltaE;Ej=Ek
    else:
        maxk=getDiffJ(i,oS.m)
        Ej = calcEk(oS,maxk)
    return maxk,Ej,maxDelatE

def innerL(i,oS):
    """
        Description:SMO算法的关键流程步骤实现
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/15 13:30
    """
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
        oS.alphas[j]=clipped(oS.alphas[j],L,H)

        update(oS,j)
        if(abs(oS.alphas[j]-alphaJold)<0.00001):
            # print('变化太小')
            return 0

        oS.alphas[i] = alphaIold + oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        update(oS, i)

        b1 = oS.b - Ei - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,i] -oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[i,j]
        b2 = oS.b - Ej - oS.labelMat[i] * (oS.alphas[i] - alphaIold) * oS.K[i,j] - oS.labelMat[j] * (
                    oS.alphas[j] - alphaJold) * oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2) / 2.0
        return 1
    else:
        return 0

def wholeSMO(dataMatIn, classLabels, C, toler, maxIter,KTup):
    """
        Description:SMO算法全部流程
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/15 13:30
    """
    oS = optStruct(np.mat(dataMatIn), np.mat(classLabels).transpose(), C, toler,KTup)  # 初始化数据结构
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
        # print("迭代次数: %d" % iter)
    return  oS.alphas,oS.b

def imageToVecor(fileName):
    imageVec=np.zeros((1,1024))
    with open( fileName) as file:
        for i in range(32):
            line=file.readline()
            for j in range(32):
                imageVec[0,i*32+j]=int(line[j])
    return imageVec

def loadImage(dirName):
    dirlist=listdir(dirName)
    labelArray=[]
    dataSetArray=np.zeros((len(dirlist),1024))
    for m in range(len(dirlist)):
        fileNames=dirlist[m].strip().split('_')
        labelArray.append(fileNames[0])
        dataSetArray[m,:]=imageToVecor(dirName+'/%s'%(dirlist[m]))
    return dataSetArray,labelArray

def OVRProcessing(dataSetArray,labelArray,i):
    labelMat=[]
    for k in range(len(dataSetArray)):
        if labelArray[k]==i:
            labelMat.append(1)
        else:
            labelMat.append(-1)
    return dataSetArray,labelMat

def SVMClassification(trainDatasetArray,tarinLabelArray,testDatasetArray,testLabelArray,C,toler,maxIter,KTup):
    trainDatasetMat=np.mat(trainDatasetArray)
    tarinLabelMat=np.mat(tarinLabelArray).transpose()
    alphas, b = wholeSMO(trainDatasetArray, tarinLabelArray, C, toler, maxIter, KTup)
    svId=np.nonzero(alphas.A>0)[0]
    svs=trainDatasetMat[svId]
    labelSv=tarinLabelMat[svId]
    m,n=np.shape(trainDatasetMat)
    errorCount=0
    for i in range(m):
        #计算核
        kernelVal=kernelTrans(svs,trainDatasetMat[i,:],KTup)
        #超平面得到预测
        predict=kernelVal.T*np.multiply(labelSv,alphas[svId])+b
        if np.sign(predict)!=np.sign(tarinLabelArray[i]):
            errorCount+=1
    trainNoAcc=float(errorCount/m)
    errorCount = 0
    testDataSetMat=np.mat(testDatasetArray)
    m,n=np.shape(testDataSetMat)
    for i in range(m):
        kernelVal=kernelTrans(svs,testDataSetMat[i,:],KTup)
        predict=kernelVal.T*np.multiply(labelSv,alphas[svId])+b
        if np.sign(predict)!=np.sign(testLabelArray[i]):
            errorCount+=1
    testNOAcc=float(errorCount/m)
    return trainNoAcc,testNOAcc

def SKRBF(trainDataSetArray,trainLabelArray,testDataSetArray,testLabelArray):
    time1 = time.time()
    svc = svm.SVC(C=100, kernel='rbf', gamma=0.0159, decision_function_shape='ovo', max_iter=-1, tol=0.0001).fit(
        trainDataSetArray, trainLabelArray)
    clresult = svc.predict(testDataSetArray)
    time2 = time.time()
    print("高斯核函数finshed in %4f secodes" % (time2 - time1))
    score = svc.score(testDataSetArray, testLabelArray)
    print('高斯核函数预测准确率%f %%' % (100 * score))
    errortCount = 0
    for i in range(len(clresult)):
        if clresult[i] != testLabelArray[i]:
            errortCount += 1
    print('高斯核函数预测错误数：', errortCount)
    print('高斯核函数预测准确率%f %%' % ((len(clresult) - errortCount) * 100 / len(clresult)))

def SKPOLY(trainDataSetArray,trainLabelArray,testDataSetArray,testLabelArray):
    """
        Description:多项式核函数的使用
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/15 13:28
    """
    time1 = time.time()
    svc = svm.SVC(kernel='poly',C=10,degree=4,gamma=0.0159,decision_function_shape='ovo',tol=0.00001).fit(
        trainDataSetArray, trainLabelArray)
    clresult = svc.predict(testDataSetArray)
    time2 = time.time()
    print("多项式核函数finshed in %4f secodes" % (time2 - time1))
    score = svc.score(testDataSetArray, testLabelArray)
    print('多项式核函数预测准确率%f %%' % (100 * score))
    errortCount = 0
    for i in range(len(clresult)):
        if clresult[i] != testLabelArray[i]:
            errortCount += 1
    print('多项式核函数预测错误数：', errortCount)
    print('多项式核函数预测准确率%f %%' % ((len(clresult) - errortCount) * 100 / len(clresult)))

def SKSigmoid(trainDataSetArray,trainLabelArray,testDataSetArray,testLabelArray):
    """
        Description:sigmoid核函数的调用
        Params:
                trainDataSetArray——训练集和测试集数据
                trainLabelArray,
                testDataSetArray,
                testLabelArray
        Return:

        Author:
                HY
        Modify:
                2019/5/15 13:27
    """
    time1 = time.time()
    svc = svm.SVC(kernel='sigmoid',gamma=0.0159,C=0.9,decision_function_shape='ovo',tol=0.00001).fit(
        trainDataSetArray, trainLabelArray)
    clresult = svc.predict(testDataSetArray)
    time2 = time.time()
    print("Sigmoid核函数finshed in %4f secodes" % (time2 - time1))
    score = svc.score(testDataSetArray, testLabelArray)
    print('Sigmoid核函数预测准确率%f %%' % (100 * score))
    errortCount = 0
    for i in range(len(clresult)):
        if clresult[i] != testLabelArray[i]:
            errortCount += 1
    print('Sigmoid核函数预测错误数：', errortCount)
    print('Sigmoid核函数预测准确率%f %%' % ((len(clresult) - errortCount) * 100 / len(clresult)))


if __name__ == '__main__':
    trainDataSetArray,trainLabelArray=loadImage('trainingDigits')
    testDataSetArray, testLabelArray = loadImage('testDigits')
    # 相当于训练了10个分类器
    # for i in range(9):
    #     trainDataSetArray, trainLabelArray=OVRProcessing(trainDataSetArray,trainLabelArray,i)
    #     testDataSetArray, testLabelArray  = OVRProcessing(testDataSetArray, testLabelArray, i)
    #     k1=1.8
    #     C=200
    #     maxIter=100
    #     time1=time.time()
    #     trainNoAcc, testNOAcc=SVMClassification(trainDataSetArray, trainLabelArray, testDataSetArray, testLabelArray,C,0.0001,maxIter,('rbf',k1))
    #     time2=time.time()
    #     print("finshed in %4f secodes"%(time2-time1))
    #     print("训练集错误率为：%f %%"% (trainNoAcc*100))
    #     print("测试集错误率为：%f %%" % (testNOAcc * 100))
    SKRBF(trainDataSetArray,trainLabelArray,testDataSetArray,testLabelArray)
    SKPOLY(trainDataSetArray,trainLabelArray,testDataSetArray,testLabelArray)
    SKSigmoid(trainDataSetArray,trainLabelArray,testDataSetArray,testLabelArray)
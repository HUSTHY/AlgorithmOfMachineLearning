#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   KernelSVM.py
@Time    :   2019/5/14 14:17
@Desc    :   1、实现了高斯核函数的SVM——采用SMO来实现对偶问题的求解
             2、可视化数据样本点、以及调用SKlearn svm.SVC来实现等高线的分类可视化
             3、使用了model_selection 中的GridSearchCV 来寻参

'''
import matplotlib.pyplot as plt
import numpy as np
import random
from  sklearn import svm
from sklearn.model_selection import GridSearchCV

def showDataSet(dataMat,labelMat):
    data_plus=[];data_minus=[]
    for i in range(len(dataMat)):
        if labelMat[i]==1:
            data_plus.append(dataMat[i])
        else:
            data_minus.append(dataMat[i])
    data_plus_np=np.array(data_plus)
    data_minus_np=np.array(data_minus)
    plt.figure(figsize=(13,7))
    plt.scatter(data_plus_np[:,0],data_plus_np[:,1],c='blue',s=30,alpha=0.8)
    plt.scatter(data_minus_np[:,0],data_minus_np[:,1],c='purple',s=30,alpha=0.8)
    plt.show()

def loadData(dataFileName):
    dataSetArray=[];labelArray=[]
    with open(dataFileName) as file:
        con=file.readlines()
    for line in con:
        line=line.strip().split('\t')
        line=[float(ele) for ele in line]
        dataSetArray.append(line[0:len(line)-1])
        labelArray.append(line[-1])
    return dataSetArray,labelArray

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
        self.K=np.mat(np.zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i]=kernelTrans(self.X,self.X[i,:],KTup)


def kernelTrans(X,A,KTup):
    m,n=np.shape(X)
    K=np.mat(np.zeros((m,1)))
    if KTup[0]=='lin':
        k=X*A.T
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
        # 步骤3计算eta
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
        # 步骤5更新alphai
        oS.alphas[i] = alphaIold + oS.labelMat[j] * oS.labelMat[i] * (alphaJold - oS.alphas[j])
        update(oS, i)

        # 计算b
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


def kernelSVM(dataSetArray,labelArray,k1):
    alphas,b=wholeSMO(dataSetArray,labelArray,100,0.00001,100,('rbf',k1)) #得到alphas和b
    dataMat=np.mat(dataSetArray);labelmat=np.mat(labelArray).transpose()
    svId=np.nonzero(alphas.A>0)[0]
    #获得支持向量
    svs=dataMat[svId]
    labelSv=labelmat[svId]
    # print('支持向量个数为：%d'% np.shape(svs)[0])
    m,n=np.shape(dataMat)
    errorCount=0
    for i in range(m):
        #k值是用支持向量来计算的
        kernelVal=kernelTrans(svs,dataMat[i,:],('rbf',k1))
        prdict=kernelVal.T*np.multiply(labelSv,alphas[svId])+b
        if np.sign(prdict)!=np.sign(labelArray[i]):
            errorCount+=1
    trainAcc=float(errorCount/m)
    print('预测错误率为：%f %% '% (errorCount*100/m) )
    dataSetArray, labelArray = loadData('RBFTestDataSet.txt')
    dataMat = np.mat(dataSetArray)
    m, n = np.shape(dataMat)
    errorCount = 0
    for i in range(m):
        kernelVal = kernelTrans(svs, dataMat[i, :], ('rbf', k1))
        predict = kernelVal.T * np.multiply(labelSv, alphas[svId]) + b
        if np.sign(predict) != np.sign(labelArray[i]):
            errorCount += 1
    print('预测错误率为：%f %% ' % (errorCount * 100 / m))
    testAcc=float(errorCount/m)
    return trainAcc,testAcc

def SkSVMWithContourLine():
    """
        Description:可视化数据样本点、以及调用SKlearn svm.SVC来实现等高线的分类可视化
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/15 13:24
    """
    dataSetArray, labelArray = loadData('RBFTrainDataSet.txt')
    dataSetArray = np.array(dataSetArray)
    labelArray = np.array(labelArray)
    colors = []
    for e in labelArray:
        if e == 1:
            colors.append('blue')
        else:
            colors.append('purple')
    x_min, x_max = dataSetArray[:, 0].min(), dataSetArray[:, 0].max()
    y_min, y_max = dataSetArray[:, 1].min(), dataSetArray[:, 1].max()
    # 构造一个坐标网络系统
    xx, yy = np.meshgrid(np.arange(x_min, x_max, (x_max - x_min) / 100), np.arange(y_min, y_max, (y_max - y_min) / 100))
    svc = svm.SVC(C=90, kernel='rbf', gamma= 2.4000000000000004,tol=0.00001,decision_function_shape='ovo',max_iter=-1).fit(dataSetArray, labelArray)
    dataSetArray1, labelArray1 = loadData('RBFTestDataSet.txt')
    print('分类预测准确率为：%f %%'%(svc.score(dataSetArray1,labelArray1)*100))
    #xx.ravel()降维度函数
    z = svc.predict(np.c_[xx.ravel(), yy.ravel()])
    z = z.reshape(np.shape(xx))
    plt.figure(figsize=(13, 7))
    #画出等高线
    plt.contourf(xx, yy, z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(dataSetArray[:, 0], dataSetArray[:, 1], s=40, c=colors)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.show()

def serachBestParametersCAndGamma(trainDataSetArray,trainLabelArray):
    """
        Description:
        Params:
                trainDataSetArray——训练集数据
                trainLabelArray——训练集标签列表
        Return:

        Author:
                HY
        Modify:
                2019/5/15 13:26
    """
    estimator=svm.SVC(kernel='rbf',decision_function_shape='ovo',tol=0.00001)
    param_test={
        'C':np.arange(90,105,1),
        'gamma':np.arange(2,2.6,0.1)
    }
    #函数的参数定义：分类器函数、参数网格、折返次数
    ParamSearch=GridSearchCV(estimator,param_grid=param_test,cv=4)
    ParamSearch.fit(trainDataSetArray,trainLabelArray)
    bestParam=ParamSearch.best_estimator_.get_params()
    print(bestParam)
    print('%f %%'%(ParamSearch.best_score_*100))
    return bestParam

if __name__ == '__main__':
   dataSetArray,labelArray=loadData('RBFTrainDataSet.txt')
   showDataSet(dataSetArray,labelArray)
   k1=1.8
   kernelSVM(dataSetArray, labelArray, k1)
   SkSVMWithContourLine()
   serachBestParametersCAndGamma(dataSetArray,labelArray)

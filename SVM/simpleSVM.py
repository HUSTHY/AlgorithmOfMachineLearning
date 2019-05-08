#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   simpleSVM.py
@Time    :   2019/5/7 15:44
@Desc    :   使用简化版的SMO算法实现了SVM算法，这里只考虑了显性可分的情况；同时给定数据做可视化显示

'''
import numpy as np
import random
import time
import matplotlib.pyplot as plt

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

def simpleSMO(dataSetArray,classLabesArray,C,toler,maxiter):
    """
        Description:简化版的SMO算法实现，在选取alpha的时候并没有使用启发式选择方法，计算很耗时
        Params:
                dataSetArray——训练数据
                classLabesArray——分类标签
                C——松弛变量
                toler——容错率
                maxiter——最大迭代次数
        Return:
                alphas——拉格朗日的系数
                b——直线截距
        Author:
                HY
        Modify:
                2019/5/8 14:36
    """
    #都使用了矩阵来进行计算
    dataSetMat=np.mat(dataSetArray);classLabesMat=np.mat(classLabesArray).T
    m,n=np.shape(dataSetMat)
    #初始化参数设置为0
    alphas=np.mat(np.zeros((m,1)));b=0
    iter=0
    while iter<maxiter:
        alphaParisChanged=0
        for i in range(m):
            #multiply对应元素相乘，维度不变
            #步骤1：计算误差Ei
            fxi=float(np.multiply(alphas,classLabesMat).T*dataSetMat*dataSetMat[i,:].T)+b
            Ei=fxi-float(classLabesMat[i])
            #在容错率内优化alpha——同时要求0<alpha<C因为只有在这个区间内才是支持向量——有KTT条件得到的
            if (Ei*classLabesMat[i]>toler and alphas[i]>0) or (Ei*classLabesMat[i]<-toler and alphas[i]<C):
                #选择一个和i不同的j 来进行alpha对的更新
                j=getDiffJ(i,m)
                # 步骤1：计算误差Ej
                fxj=float(np.multiply(alphas,classLabesMat).T*dataSetMat*dataSetMat[j,:].T)+b
                Ej=fxj-float(classLabesMat[j])
                #保存alpha的值，后面会用到，因为这里对象的引用，有深拷贝、浅拷贝的问题
                alphaIold=alphas[i].copy()
                alphaJold=alphas[j].copy()
                #步骤2：计算alpha的上下界
                if classLabesMat[i]==classLabesMat[j]:
                    L=max(0,alphaJold+alphaIold-C)
                    H=min(C,alphaJold+alphaIold)
                else:
                    L=max(0,alphaJold-alphaIold)
                    H=min(C,C+alphaJold-alphaIold)
                if L==H:
                    #continue跳出本次循环
                    continue
                #步骤3：计算eta
                eta=dataSetMat[j,:] * dataSetMat[j,:].T+dataSetMat[i,:]*dataSetMat[i,:].T-2.0*dataSetMat[j,:]*dataSetMat[i,:].T
                if eta<=0:
                    continue
                #步骤4：更新alphaJ
                alphas[j]=alphaJold+classLabesMat[j]*(Ei-Ej)/eta
                #步骤5：修剪alphaJ
                alphas[j]=clipped(alphas[j],L,H)
                # 步骤6： 更新alphaI
                alphas[i]=alphaIold+classLabesMat[j,:]*classLabesMat[i]*(alphaJold-alphas[j])
                # 步骤7：计算b1和b2
                b1=b-Ei-classLabesMat[i]*(alphas[i]-alphaIold)*dataSetMat[i,:]*dataSetMat[i,:].T-classLabesMat[j]*(alphas[j]-alphaJold)*dataSetMat[j,:]*dataSetMat[i,:].T
                b2=b-Ej-classLabesMat[i]*(alphas[i]-alphaIold)*dataSetMat[i,:]*dataSetMat[j,:].T-classLabesMat[j]*(alphas[j]-alphaJold)*dataSetMat[j,:]*dataSetMat[j,:].T
                # 步骤8：更新b
                if 0<alphas[i]<C:
                    b=b1
                elif 0<alphas[j]<C:
                    b=b2
                else:
                    b=(b1+b2)/2.0
                alphaParisChanged+=1
        if alphaParisChanged==0:
            iter+=1
        else:
            iter=0
    print('iter',iter)
    return alphas,b

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
            print(x)
            print(y)
            # x=dataSetMat[i][:,0]
            # y=dataSetMat[i][:,1]
            plt.scatter([x],[y],s=250,c='none',alpha=0.8,linewidths=1.0,edgecolors='black')
    plt.show()


if __name__ == '__main__':
    dataSetArray,labelArray=loadData('testSet.txt')
    time1=time.time()
    alphas,b=simpleSMO(dataSetArray,labelArray,0.6,0.001,40)
    time2 =time.time()
    print('finshed in %d seconds'%(time2-time1))
    wVec=calculateWVec(alphas,dataSetArray,labelArray)
    showDatasAndClassificatonResult(dataSetArray,labelArray,wVec,b,alphas)

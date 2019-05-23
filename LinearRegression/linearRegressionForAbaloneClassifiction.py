#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   linearRegressionForAbaloneClassifiction.py
@Time    :   2019/5/21 16:31
@Desc    :

'''
import  numpy as np
from sklearn.model_selection import StratifiedKFold#分层交叉验证
from sklearn.model_selection import KFold #不分层，简单的交叉验证
import time

def loadDataSet(fileName):
    x=[];y=[]
    with open(fileName) as file:
        con = file.readlines()
    for ele in con:
        currLine=ele.strip().split('\t')
        arr=[]
        for i in range(len(currLine)-1):
            arr.append(float(currLine[i]))
        x.append(arr)
        y.append(float(currLine[-1]))
        if len(x)==500:#控制数据量，全部的数据对于局部加权线性回归很大
            break
    return x,y

def lwlrComputeWeights(testPoint,x,y,k):
    """
        Description:局部加权线性回归求解回归系数
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/22 18:00
    """
    m=len(y)
    xMat=np.mat(x)
    yMat=np.mat(y).T
    weihts=np.eye(m)
    for i in range(m):
        diffMat=testPoint-xMat[i,:]
        #计算得到回归系数
        weihts[i,i]=np.exp(diffMat*diffMat.T/(-2*k*k))
    if np.linalg.det(xMat.T*weihts*xMat)==0:
        print(np.linalg.det(xMat.T * weihts * xMat))
        print('矩阵XWX是不可逆矩阵！')
        time.sleep(2)
        return
    else:
        ttt = (xMat.T * (weihts * xMat)).I
        ws = (xMat.T * (weihts * xMat)).I * (xMat.T * (weihts * yMat))
        return testPoint * ws


def lwlrComputeResult(x_cordArr, y_cordArr,k):
    m=len(x_cordArr)
    yPredictMat=np.zeros(m)
    for i in range(m):
        yPredictMat[i]=lwlrComputeWeights(x_cordArr[i],x_cordArr,y_cordArr,k)
    return  yPredictMat

def calculateError(y_predict,y):
    """
        Description:计算预测值和真实值之间的误差
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/22 17:59
    """
    m=len(y_predict)
    error=(y_predict-y)**2
    return sum(error)/m

def abaloneClassification(x,y):
    x=np.array(x);y=np.array(y)
    n_sp=5
    # skf = StratifiedKFold(n_splits=n_sp, random_state=11)
    #不分层交叉验证
    kf=KFold(n_splits=n_sp,random_state=11)
    k=[50,10,1,0.1]
    for ele in k:
        train_errors = 0;valid_errors = 0
        # for trainIndex, validIndex in skf.split(x, y):
        i=0
        for trainIndex, validIndex in kf.split(x, y):
            x_train, x_valid = x[trainIndex], x[validIndex]
            y_train, y_valid = y[trainIndex], y[validIndex]
            y_train_predict = lwlrComputeResult(x_train, y_train, ele)
            y_valid_predict = lwlrComputeResult(x_valid, y_valid, ele)
            train_errors+=calculateError(y_train_predict,y_train)
            valid_errors+=calculateError(y_valid_predict,y_valid)
            i+=1
        print(i)
        print('采用局部加权预测鲍鱼的年龄，采用5折交叉验证，核K和均方误差的关系：')
        print('k==%f时，训练集的均方误差：%0.4f，验证集中的均方误差：%0.4f'%(ele,train_errors/n_sp,valid_errors/n_sp))

if __name__ == '__main__':
    x, y=loadDataSet('abalone.txt')
    # x, y=loadDataSet('testDataSet.txt')
    time1=time.time()
    abaloneClassification( x, y)
    time2=time.time()
    print('Finshed in %0.4f seconds！'%(time2-time1))
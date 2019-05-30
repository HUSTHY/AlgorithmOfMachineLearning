#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   randomForestClassifier.py
@Time    :   2019/5/29 12:50
@Desc    :   1、使用随机森林分类器算法对数字进行识别、画学习曲线和验证曲线
             2、使用随机森林回归器算法对鲍鱼年龄进行预测。画学习曲线和验证曲线

'''
import numpy as np
from os import listdir
from sklearn.metrics import mean_squared_error#均方误差
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor


import time
def dealFileToVec(fileName):
    """
        Description:把图片信息转化为向量
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/29 21:00
    """
    vec=np.zeros((1,1024))
    with open(fileName) as file:
        for i in range(32):
            line = file.readline()
            for j in range(len(line)-1):
                vec[0, i * 32 + j] = int(line[j])
    return vec

def loadDataSet(fileDir):
    dataSet=[];label=[]
    dir_list=listdir(fileDir)#获取文件夹下的所有目录和文件名
    m=len(dir_list)
    dataSet=np.zeros((m,1024))
    for i in range(m):
        fileName = dir_list[i]
        dataSet[i,:]=dealFileToVec(fileDir+'/'+fileName)
        label.append(int(fileName.strip().split('_')[0]))
    return dataSet,label

def plot_learning_curve_classification(algorithm,trainDataSet,trainLabel,testDataSet,testLabel):
    """
        Description：学习曲线，训练样本越多——准确率和训练集规模的一个关系
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/29 14:02
    """
    train_score=[];test_score=[]
    for i in range(1,len(trainDataSet)+1,100):
        algorithm.fit(trainDataSet[0:i],trainLabel[0:i])
        #得出预测准确率
        train_score.append(algorithm.score(trainDataSet[0:i],trainLabel[0:i]))
        test_score.append(algorithm.score(testDataSet,testLabel))
    x=[i for i in range(1,len(trainDataSet)+1,100)]
    plt.figure(figsize=(18,14))
    plt.plot(x,np.sqrt(train_score),c='b',alpha=0.8,label='train')
    plt.plot(x,np.sqrt(test_score),c='r',alpha=0.8, label='test')
    plt.ylabel('Accuracy')
    plt.xlabel('DataSet count')
    plt.legend()
    plt.show()

def plot_validation_cure_classification(trainDataSet,trainLabel,testDataSet,testLabel,n_estimators_options,max_depth_options):
    """
        Description:验证曲线，得到不同参数是模型的性能
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/29 21:01
    """
    plt.figure(figsize=(13,8))
    best_test_score=0;best_n_estimators=0
    train_score = [];test_score = []
    time1 = time.time()
    for param in n_estimators_options:
        algorithm = RandomForestClassifier(n_estimators=param, oob_score=True,random_state=9)
        train_score.append(algorithm.fit(trainDataSet, trainLabel).score(trainDataSet, trainLabel))
        testscore = algorithm.fit(trainDataSet, trainLabel).score(testDataSet, testLabel)
        test_score.append(testscore)
        if best_test_score < testscore:
            best_test_score = testscore
            best_n_estimators = param
    time2 = time.time()
    print('finished in %.4f' % (time2 - time1))
    plt.plot(n_estimators_options, train_score, c='b', label='train')
    plt.plot(n_estimators_options, test_score, c='r', label='test')
    plt.ylabel('SOCRE')
    plt.xlabel('n_estimators count')
    plt.title('the relationship between n_estimators and score ')
    plt.legend()
    plt.show()
    print('best_test_score: %.4f, best_n_estimators: %d'%(best_test_score, best_n_estimators))

    del train_score[:];del test_score[:]
    best_test_score = 0;best_max_depth = 0
    time1 = time.time()
    for param in max_depth_options:
        algorithm = RandomForestClassifier(n_estimators=best_n_estimators,max_depth=param, oob_score=True,random_state=9)
        train_score.append(algorithm.fit(trainDataSet, trainLabel).score(trainDataSet, trainLabel))
        testscore = algorithm.fit(trainDataSet, trainLabel).score(testDataSet, testLabel)
        test_score.append(testscore)
        if best_test_score < testscore:
            best_test_score = testscore
            best_max_depth = param
    time2 = time.time()
    print('finished in %.4f' % (time2 - time1))
    plt.plot(max_depth_options, train_score, c='b', label='train')
    plt.plot(max_depth_options, test_score, c='r', label='test')
    plt.ylabel('SOCRE')
    plt.xlabel('max_depth count')
    plt.title('the relationship between max_depth and score ')
    plt.legend()
    plt.show()
    print('best_test_score: %.4f, best_max_depth: %d'%(best_test_score, best_max_depth))

    del train_score[:];
    del test_score[:]
    best_test_score = 0;best_max_depth = 0;best_n_estimators=0
    estimators=[];depth=[]
    time1 = time.time()
    for n_estimator in n_estimators_options:
        for max_depth in max_depth_options:
            algorithm = RandomForestClassifier(n_estimators=n_estimator,max_depth=max_depth, oob_score=True)
            train_score.append(algorithm.fit(trainDataSet, trainLabel).score(trainDataSet, trainLabel))
            testscore = algorithm.fit(trainDataSet, trainLabel).score(testDataSet, testLabel)
            test_score.append(testscore)
            estimators.append(n_estimator)
            depth.append(max_depth)
            if best_test_score < testscore:
                best_test_score = testscore
                best_max_depth = max_depth
                best_n_estimators=n_estimator
    time2 = time.time()
    print('finished in %.4f' % (time2 - time1))
    ax=plt.axes(projection='3d')
    ax.plot3D(estimators,depth,train_score,c='b',label='train')
    ax.plot3D(estimators, depth, test_score, c='r', label='test')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('max_depth')
    ax.set_zlabel('score')
    plt.title('the relationship between max_depth ,n_estimators and score ')
    plt.show()
    print('best_test_score: %.4f, best_max_depth: %d ,best_n_estimators:%d' % (best_test_score, best_max_depth,best_n_estimators))

def loadabaloneDataSet(fileName):
    dataSet=[];label=[]
    with open(fileName) as file:
        lines=file.readlines()
    for i in range(len(lines)):
        currLine=lines[i].strip().split('\t')
        arr=[]
        for j in range(len(currLine)-1):
            arr.append(float(currLine[j]))
        dataSet.append(arr)
        label.append(float(currLine[-1]))
    return dataSet,label

def plot_learnging_curve_regression(trainDataSet,trainLabel,validDataSet,validLabel,testDataSet,testLabel):
    rfr=RandomForestRegressor(n_estimators=50,oob_score=True,random_state=5)
    train_score=[];valid_score=[];test_score=[]
    for i in range(1,len(trainDataSet)+1,100):
        rfr.fit(trainDataSet[0:i],trainLabel[0:i])
        #返回的是决定系数
        train_score.append(rfr.score(trainDataSet,trainLabel))
        valid_score.append(rfr.score(validDataSet,validLabel))
        test_score.append(rfr.score(testDataSet,testLabel))
    x = [i for i in range(1, len(trainDataSet) + 1, 100)]
    plt.figure(figsize=(18, 14))
    plt.plot(x,train_score,c='b',label='train')
    plt.plot(x,test_score,c='r',label='test')
    plt.plot(x,valid_score,c='purple',label='valid')

    plt.title('learning_curve')
    plt.xlabel('dataSet count')
    plt.ylabel('coefficient of determination')
    plt.legend()
    plt.show()

def plot_validation_curve_regression(trainDataSet,trainLabel,validDataSet,validLabel,testDataSet,testLabel,n_estimators_options,max_depth_options):
    train_score = [];
    valid_score = [];
    test_score = []
    for n_estimator in n_estimators_options:
        rfr=RandomForestRegressor(n_estimators=n_estimator,oob_score=True,random_state=5)
        rfr.fit(trainDataSet,trainLabel)
        train_score.append(rfr.score(trainDataSet,trainLabel))
        valid_score.append(rfr.score(validDataSet,validLabel))
        test_score.append(rfr.score(testDataSet,testLabel))
    plt.figure(figsize=(13,8))
    plt.plot(n_estimators_options,train_score,c='b')
    plt.plot(n_estimators_options,valid_score,c='r')
    plt.plot(n_estimators_options,test_score,c='purple')
    plt.show()

    del train_score[:];del valid_score[:]; del test_score[:]
    for max_depth in max_depth_options:
        rfr = RandomForestRegressor(n_estimators=200, oob_score=True, random_state=5,max_depth=max_depth)
        rfr.fit(trainDataSet, trainLabel)
        train_score.append(rfr.score(trainDataSet, trainLabel))
        valid_score.append(rfr.score(validDataSet, validLabel))
        test_score.append(rfr.score(testDataSet, testLabel))
    plt.figure(figsize=(13, 8))
    plt.plot(max_depth_options, train_score, c='b')
    plt.plot(max_depth_options, valid_score, c='r')
    plt.plot(max_depth_options, test_score, c='purple')
    plt.show()

if __name__ == '__main__':
    trainDataSet, trainLabel=loadDataSet('trainingDigits')
    testDataSet, testLabel=loadDataSet('testDigits')
    #rf随机森林分类器——学习曲线
    rf=RandomForestClassifier(n_estimators=20)
    time1=time.time()
    plot_learning_curve_classification(rf,trainDataSet, trainLabel,testDataSet, testLabel)
    time2=time.time()
    print('finished in %.4f'%(time2-time1))

    #RF分类器调参数——验证曲线
    n_estimators_options = list(range(265,275,1))
    max_depth_options=list(range(735,745,1))
    time1=time.time()
    plot_validation_cure_classification(trainDataSet, trainLabel,testDataSet, testLabel,n_estimators_options,max_depth_options)
    time2=time.time()
    print('finished in %.4f' % (time2 - time1))
    rf = RandomForestClassifier(n_estimators=270,max_depth=741,random_state=9)
    rf.fit(trainDataSet, trainLabel)
    score=rf.score(testDataSet,testLabel)
    print('score: ',score)
    l_test_predict=rf.predict(testDataSet)
    errorcount=0
    for i in range(len(l_test_predict)):
        if l_test_predict[i]!=testLabel[i]:
            errorcount+=1
    print(errorcount)
    acc=(len(testLabel)-errorcount)/len(testLabel)
    print(acc)

    #单棵决策树
    DTC=DecisionTreeClassifier()
    DTC.fit(trainDataSet, trainLabel)
    score = DTC.score(testDataSet,testLabel)
    print('DecisionTreeClassifier score: ',score)

    #KNN算法
    KNN=KNeighborsClassifier(n_neighbors=3)
    score=KNN.fit(trainDataSet, trainLabel).score(testDataSet,testLabel)
    print('KNeighborsClassifier score: ',score)
    dataSet, label=loadabaloneDataSet('abalone.txt')
    X,testDataSet,Y,testLabel=train_test_split( dataSet, label,test_size=0.2,random_state=1)
    trainDataSet,validDataSet,trainLabel,validLabel=train_test_split(X,Y,test_size=0.2,random_state=1)
    # plot_learnging_curve_regression(trainDataSet,trainLabel,validDataSet,validLabel,testDataSet,testLabel)
    n_estimators_options=list(range(200,201,2))
    max_depth_options=list(range(10,121,10))
    plot_validation_curve_regression(trainDataSet, trainLabel, validDataSet, validLabel, testDataSet, testLabel,n_estimators_options, max_depth_options)


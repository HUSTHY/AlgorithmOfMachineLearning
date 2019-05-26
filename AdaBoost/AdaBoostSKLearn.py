#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   AdaBoostSKLearn.py
@Time    :   2019/5/17 10:28
@Desc    :   基于其他的分类方法实现的一个adaboost算法——其他的分类算法如：LR、SVM、Bayes等等

'''
import numpy as np
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def myOwnAdaBoost(trainDataSet,trainLabel,testDataSet,testLabel,clfs,maxIter):
    trainDataSetMat=np.mat(trainDataSet)
    trainLabelMat=np.mat(trainLabel).transpose()
    testDataSetMat=np.mat(testDataSet)
    alphas=[]
    Classifiers=[]
    m,n=np.shape(trainDataSetMat)
    k=np.shape(testDataSetMat)[0]
    D=np.ones((m,1))/m
    finalTraRes=np.zeros((m,1))
    testFinalRes=np.zeros((k,1))
    for i in range(maxIter):
        #计算需要把多维降为一维
        sw=np.ravel(D)
        minError, BTrainRes, BTestRes=minErrorClassification(trainDataSet, trainLabel,testDataSet,testLabel,clfs,sw)
        #计算需要把一维升为多维
        BTrainRes=BTrainRes.reshape((len(BTrainRes),1))
        BTestRes=BTestRes.reshape((len(BTestRes),1))
        Classifiers.append(BTestRes)
        errorMat=np.ones((m,1))
        errorMat[BTrainRes==trainLabelMat]=0
        errorWeigs=float(np.dot(errorMat.T,D))
        if errorWeigs==0:
            print('errorWeigs:误差为0')
            break
        alpha=0.5*np.log((1-errorWeigs)/errorWeigs)
        alphas.append(alpha)
        expVal=np.exp(-1*alpha*np.multiply(BTrainRes,trainLabelMat))
        D=np.multiply(D,expVal)
        D=D/sum(D)
        finalTraRes+=alpha*BTrainRes

        testFinalRes+=alpha*BTestRes
        if (np.sign(finalTraRes)==trainLabelMat).all():
            print('训练集中分类完全正确')
            break
    return alphas,Classifiers,np.sign(testFinalRes)
def minErrorClassification(trainDataSet, trainLabel,testDataSet,testLabel,clfs,D):
    m=np.shape(trainDataSet)[0]
    BTestRes=BTrainRes=np.empty((m,1))
    minError=float('inf')
    for clf in clfs:
        clf.fit(trainDataSet,trainLabel,sample_weight=D)
        error=1-clf.score(trainDataSet,trainLabel)
        testPreRes=clf.predict(testDataSet)
        trainPreRes=clf.predict(trainDataSet)
        print('训练集中单个分类器分类准确率：%f'%(clf.score(trainDataSet,trainLabel)))
        # print('测试集中单个分类器分类准确率：%f' % (clf.score(testDataSet, testLabel)))
        # print('error:%f'%error)
        if minError>error:
            minError=error
            BTrainRes=trainPreRes.copy()
            BTestRes=testPreRes.copy()
    print('minError:',minError)
    return minError,BTrainRes,BTestRes

def loadDataSet(fileName):
    dataSet=[];label=[]
    with open(fileName) as file:
        con=file.readlines()
    for ele in con:
        line=ele.strip().split('\t')
        lineArr=[]
        for l in range(len(line)-1):
            lineArr.append(float(line[l]))
        dataSet.append(lineArr)
        label.append(float(line[-1]))
    label=[-1.0 if ele==0 else ele for ele in label]
    return dataSet,label
if __name__ == '__main__':
    trainDataSet, trainLabel=loadDataSet('HorseData\horseColicTraining.txt')
    testDataSet,testLabel=loadDataSet('HorseData\horseColicTest.txt')
    maxIter=20
    clf0=SVC(C=500,kernel='linear',max_iter=-1,gamma=0.001)
    clf1 = SVC(C=400, kernel='linear', max_iter=-1, gamma=0.01)
    clf2 = SVC(C=350, kernel='linear', max_iter=-1, gamma=0.1)

    # clf1=DecisionTreeClassifier(max_depth=2)
    # clf2=LogisticRegression(penalty='l2',max_iter=2000,C=0.0074,solver='saga')
    clf3 = DecisionTreeClassifier(max_depth=2)
    clf4 = DecisionTreeClassifier(max_depth=3)
    clf5 = DecisionTreeClassifier(max_depth=4)
    clfs=[clf0,clf1,clf2]
    # clfs = [clf3, clf4, clf5]
    alphas,Classifiers,testFinalRes=myOwnAdaBoost(trainDataSet, trainLabel,testDataSet,testLabel,clfs,maxIter)
    errorCount=0
    for i in range(len(testLabel)):
        if int(testLabel[i])!=int(testFinalRes[i,:]):
            errorCount+=1
    print('测试集分类准确率为：%f %%'%((len(testLabel) - errorCount) * 100 /len(testLabel)))
    # clf1.fit(trainDataSet,trainLabel)
    # clf1.predict(testDataSet)
    # print(clf1.score(testDataSet,testLabel))
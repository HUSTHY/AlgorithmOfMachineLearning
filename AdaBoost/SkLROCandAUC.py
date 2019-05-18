#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   SkLROCandAUC.py
@Time    :   2019/5/18 21:46
@Desc    :   Sklearn包中的model_selection做交叉验证；metrics做性能指标评估；ensemble中的adaBoost做分类器
'''
from sklearn.model_selection import StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
import numpy as np

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

def classificationDataSet(dataSet, label):
    dataSet=np.array(dataSet);label=np.array(label)
    #分层k折交叉验证数据划分
    skf = StratifiedKFold(n_splits=5)
    Dtree=DecisionTreeClassifier(max_depth=2)
    #Adaboost强可学习分类器
    Ada=AdaBoostClassifier(base_estimator=Dtree,n_estimators=60,learning_rate=0.8)
    FPRs=[];TPRs=[];roc_AUCs=[]
    for trainIndex,testIndex in skf.split(dataSet,label):
        trainDataSet,testDataSet=dataSet[trainIndex],dataSet[testIndex]
        trainLabel,testLabel=label[trainIndex],label[testIndex]
        Ada.fit(trainDataSet,trainLabel)
        #预测分类得分：scores把标签从小到大来排列，预测打分
        scores=Ada.predict_proba(testDataSet)
        #计算FPR、TPR等
        FPR,TPR,thresholds=roc_curve(testLabel,scores[:,1])
        roc_AUC=auc(FPR,TPR)
        FPRs.append(FPR)
        TPRs.append(TPR)
        roc_AUCs.append(roc_AUC)
    return FPRs,TPRs,roc_AUCs

def showRoc(FPRs,TPRs,roc_AUCs):
    colors=['b','r','y','g','c']
    plt.figure(figsize=(10,6))
    for i in range(len(FPRs)):
        plt.plot(FPRs[i],TPRs[i],c=colors[i],label='AUC is :%0.4f'%roc_AUCs[i])
    plt.plot([0,1],[0,1],c='black',lw=2)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    dataSet, label = loadDataSet('HorseData\horseColicTraining.txt')
    FPRs,TPRs,roc_AUCs=classificationDataSet(dataSet, label)
    showRoc(FPRs,TPRs,roc_AUCs)
    # testdataSet, testlabel = loadDataSet('HorseData\horseColicTest.txt')
    # Dtree = DecisionTreeClassifier(max_depth=2)
    # # Adaboost强可学习分类器
    # Ada = AdaBoostClassifier(base_estimator=Dtree, n_estimators=60, learning_rate=0.8)
    # Ada.fit(np.array(dataSet),np.array(label))
    # # 预测分类得分：scores把标签从小到大来排列，预测打分
    # scores = Ada.predict_proba(np.array(testdataSet))
    # # 计算FPR、TPR等
    # FPR, TPR, thresholds = roc_curve(np.array(testlabel), scores[:, 1])
    # roc_AUC = auc(FPR, TPR)
    # FPRss=[];TPRss=[];roc_AUCss=[]
    # FPRss.append(FPR)
    # TPRss.append(TPR)
    # roc_AUCss.append(roc_AUC)
    # showRoc(FPRss, TPRss, roc_AUCss)
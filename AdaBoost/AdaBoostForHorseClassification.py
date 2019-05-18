#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   AdaBoostForHorseClassification.py
@Time    :   2019/5/17 17:43
@Desc    :  调用sklearn的函数来实现马匹死亡率的预测
'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
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

if __name__ == '__main__':
    trainDataSet, trainLabel = loadDataSet('HorseData\horseColicTraining.txt')
    testDataSet, testLabel = loadDataSet('HorseData\horseColicTest.txt')
    clf=SVC(C=300,kernel='rbf',max_iter=-1,gamma=0.001)
    Dtree=DecisionTreeClassifier(max_depth=2)
    AdBC=AdaBoostClassifier(base_estimator=Dtree,n_estimators=50,learning_rate=0.701,algorithm='SAMME')
    AdBC.fit(trainDataSet, trainLabel)
    result=AdBC.predict(trainDataSet)
    print(AdBC.score(trainDataSet,trainLabel))
    accurat=sum(result==trainLabel)/len(trainLabel)
    print(accurat)
    testResult=AdBC.predict(testDataSet)
    print(AdBC.score(testDataSet,testLabel))
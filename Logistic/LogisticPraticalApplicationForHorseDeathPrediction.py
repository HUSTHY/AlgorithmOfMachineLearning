import numpy as np
import random
import time
from sklearn.linear_model import LogisticRegression as LR
def dataPrePrcossing(dataFileName):
    """
        Description:数据文件中获取数据
        Params:
                dataFileName——数据文件名称(包含路劲)
        Return:
                trainDataSet——训练集
                trainLabel——分类标签
                pureHorseTotalTrainDataSet——训练集合分类标签集
        Author:
                HY
        Modify:
                2019/5/4 18:15
    """
    trainDataSet=[];trainLabel=[];pureHorseTotalTrainDataSet=[]
    with open(dataFileName,'r') as file:
        content=file.readlines()
    for con in content:
        datas=con.strip().split(' ')
        col=0
        for data in datas:
            if col==2:
                datas.remove(data)
            col+=1
        datas=[0.000 if data=='?'else float(data) for data in datas]
        trainDataSet.append(datas[0:(len(datas)-6)])
        trainLabel.append(datas[len(datas)-6])
        pureHorseTotalTrainDataSet.append(datas[0:(len(datas)-5)])
    trainLabel=[0.000 if ele!=1.0 else ele for ele in trainLabel]
    return trainDataSet,trainLabel,pureHorseTotalTrainDataSet

def storeDataSet(datas,fileName):
    with open(fileName,'w') as file:
        for data in datas:
            for d in data:
                file.write(str(d))
                file.write('\t')
            if data!=datas[-1]:
                file.write('\n')


def sigmoid(X):

    """
      Description:计算给定数据的值sigmoid函数
      Params:
            X——数据
      Return:
      Author:
            HY
      Modify:
            2019/5/1 17:28
    """

    return 1.0/(1+np.exp(-X))

def calculateAccurate(dataMat,labels,weight):

    """
      Description:根据回归系数判断分类的准确率
      Params:
             dataMat——数据集
             labels——分类集
             weight——回归系数
      Return:
             accurte——分类准确率
      Author:
            HY
      Modify:
            2019/5/1 17:29
    """
    t=np.dot(dataMat,weight)
    result=sigmoid(t)
    errcount=0
    for i in range(len(labels)):
        if int(labels[i])==1:
            if result[i]<0.5:
                errcount+=1
        else:
            if (1-result[i])<0.5:
                errcount+=1
    accurte=float((len(labels)-errcount)/len(labels))
    return accurte

def gradAscent(dataMat,labels):

    """
      Description:梯度上升法来计算参数向量
      Params:
             dataMat——训练集矩阵
             labels——分类集
      Return:
            bestWeights.getA()——最佳参数
            maxCycle——最佳迭代次数
      Author:
            HY
      Modify:
            2019/5/1 17:24
    """
    dataMatrix=np.mat(dataMat)
    labelMat=np.mat(labels).transpose()
    m,n=np.shape(dataMatrix)
    bestWeights=np.ones(n)
    aplha=0.001
    biggestAccurate=0
    # maxCycle=0
    # for i in range(1,40002,10000):
    weights=np.ones((n,1))
    cycle=0
    while cycle<30001:
            #使用梯度上升矢量公式
        r=sigmoid(dataMatrix*weights)
        h=labelMat-r
        temp=aplha*dataMatrix.transpose()*h
        weights=weights+temp
        cycle+=1
    accurte=calculateAccurate(dataMatrix,labels,weights)
        # if biggestAccurate<accurte:
        #     biggestAccurate=accurte
        #     maxCycle=i
        #     bestWeights=weights
    # print('梯度上升法：训练集中————最佳迭代次数:%d——分类准确率%d%%'%(maxCycle,biggestAccurate*100))
    print('梯度上升法：训练集中分类准确率%d%%' % (accurte*100))
    return weights


def gradScochasticAscent(dataMat,labels):

    """
      Description:梯度上升法来计算参数向量
      Params:
             dataMat——训练集矩阵
             labels——分类集
      Return:
            bestWeights.getA()——最佳参数
            maxCycle——最佳迭代次数
      Author:
            HY
      Modify:
            2019/5/1 17:24
    """
    dataMatrix=np.array(dataMat)
    m,n=np.shape(dataMatrix)
    bestWeights=np.ones(n)
    weights = np.ones(n)
    # biggestAccurate=0
    for j in range(435):
        dataIndex=list(range(m))
        for i in range(m):
            randomIndex=int(random.uniform(0,len(dataIndex)))#选择随机的一个样本来更新回归系数
            aplha=1/(1+i+j)+0.001
            h=sigmoid(sum(dataMatrix[randomIndex]*weights))
            temp=labels[randomIndex]-h
            weights=weights+aplha*dataMatrix[randomIndex].transpose()*temp
            del(dataIndex[randomIndex])
    accurte=calculateAccurate(dataMatrix,labels,weights)
    print('随机梯度上升法中：分类准确率%d%%' % (accurte*100))
    return weights

def myOwnLogisitcClassificationResult(trainDataSet,trainLabel,testDataSet,testLabel,functionType):
    """
        Description:使用自己写的分类器进行分类的预测
        Params:
                trainDataSet——训练集数据
                trainLabel——训练集分类标签集
                testDataSet——测试集数据
                testLabel——测试集分类标签集
                functionType——0：批量梯度；1：随机梯度
        Return:

        Author:
                HY
        Modify:
                2019/5/4 18:57
    """
    if functionType==0:
        weights=gradAscent(trainDataSet,trainLabel)
        result=calculateAccurate(testDataSet,testLabel,weights)
        # result = calculateAccurate(trainDataSet, trainLabel, weights)
        print('梯度上升法分类结果的准确率：%f %%'%(result*100))
    else:
        weights=gradScochasticAscent(trainDataSet,trainLabel)
        result = calculateAccurate(testDataSet, testLabel, weights)
        # result = calculateAccurate(trainDataSet, trainLabel, weights)
        print('随机梯度上升法分类结果的准确率：%f %%' % (result * 100))

def SKLLogisticRegressionClassificaitonResult(trainDataSet,trainLabel,testDataSet,testLabel):
    """
        Description:使用SKL库中的linear_model.LogisticRegression来进行分类预测
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/4 23:24
    """
    # LRClassifier=LR(penalty='l2',solver='liblinear',max_iter=50)
    LRClassifier = LR(penalty='l2', solver='sag', max_iter=2000,C=0.00741)
    LRClassifier.fit(trainDataSet,trainLabel)
    result=LRClassifier.score(testDataSet,testLabel)
    print('SKL库中LogisticRegression算法正确率为：%f %%'%(result*100))



if __name__ == '__main__':
    trainDataSet,trainLabel,pureHorseTotalTrainDataSet=dataPrePrcossing('horseData\PrimaryData\horse-colic.data')
    testDataSet, testLabel, pureHorseTotalTestDataSet = dataPrePrcossing('horseData\PrimaryData\horse-colic.test')
    storeDataSet(pureHorseTotalTrainDataSet,'horseData\pureData\horse_colic_Train.data')
    storeDataSet(pureHorseTotalTrainDataSet,'horseData\pureData\horse_colic_Test.data')
    time1=time.time()
    myOwnLogisitcClassificationResult(trainDataSet,trainLabel,testDataSet,testLabel,0)
    time2=time.time()
    print('Finshed in %f'%(time2-time1))
    time1 = time.time()
    myOwnLogisitcClassificationResult(trainDataSet, trainLabel, testDataSet, testLabel, 1)
    time2 = time.time()
    print('Finshed in %f' % (time2 - time1))
    time1 = time.time()
    SKLLogisticRegressionClassificaitonResult(trainDataSet,trainLabel,testDataSet,testLabel)
    time2 = time.time()
    print('Finshed in %f' % (time2 - time1))
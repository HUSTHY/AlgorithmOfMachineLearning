import numpy as np
import matplotlib.pyplot as plt
import time
import random
def loadData():
    fileName='testDataSet.txt'
    labels=[];dataMat=[]
    with open(fileName)as file:
        content=file.readlines()
    for c in content:
        l=c.strip().split('\t')
        dataMat.append([1.0,float(l[0]),float(l[1])])
        labels.append(int(l[2]))
    return dataMat,labels

def plotDataWeights(weights):
    dataMat,labels=loadData()
    dataArr=np.array(dataMat)
    x1=[];y1=[]
    x2=[];y2=[]
    for i in range(len(labels)):
        if labels[i]==1:
            x1.append(dataArr[i,1]);y1.append(dataArr[i,2])
        else:
            x2.append(dataArr[i,1]);y2.append(dataArr[i,2])
    plt.figure(figsize=(8,6))
    plt.scatter(x1,y1,color='red',s=20,alpha=0.5)
    plt.scatter(x2,y2,color='blue',s=20,alpha=0.5)
    x=np.arange(-3,3,0.01)
    #z=w0*x0+w1*x1+w2*x2 令z=0得 然后推测Y值
    y=(-weights[0]-weights[1]*x)/weights[2]
    plt.plot(x,y,color='black',alpha=0.5)
    plt.title('DataSet')
    plt.xlabel('X');plt.ylabel('Y')
    plt.show()

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
    labelMat=np.mat(labels).transpose()#转置
    m,n=np.shape(dataMatrix)
    bestWeights=np.ones(n)
    weights_array=np.array([])
    aplha=0.001
    biggestAccurate=0
    for i in range(1,10000,50):
        weights=np.ones((n,1))
        cycle=0
        while cycle<i:
            #使用梯度上升矢量公式
            temp=aplha*dataMatrix.transpose()*(labelMat-sigmoid(dataMatrix*weights))
            weights=weights+temp
            cycle+=1
            if i==6001:
                weights_array=np.append(weights_array,weights)
        accurte=calculateAccurate(dataMatrix,labels,weights)
        if biggestAccurate<accurte:
            biggestAccurate=accurte
            maxCycle=i
            bestWeights=np.array(weights)
    print("********************************************")
    print(bestWeights)
    print('最佳迭代次数:%d——分类准确率%d%%'%(maxCycle,biggestAccurate*100))
    weights_array=weights_array.reshape(6001,n)

    return bestWeights,maxCycle,weights_array



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
    biggestAccurate=0
    weights_array=np.array([])
    for j in range(1000):
        dataIndex=list(range(m))
        weights=np.ones(n)
        for i in range(m):
            randomIndex=int(random.uniform(0,len(dataIndex)))
            aplha=1/(1+i+j)+0.001
            h=sigmoid(sum(dataMatrix[randomIndex]*weights))
            temp=labels[randomIndex]-h
            weights=weights+aplha*dataMatrix[randomIndex].transpose()*temp
            weights_array=np.append(weights_array,weights,axis=0)#axis=0矩阵列数相同，竖着加；axis=1行数相同，横着加
            del(dataIndex[randomIndex])
        accurte=calculateAccurate(dataMatrix,labels,weights)
        if biggestAccurate<accurte:
            biggestAccurate=accurte
            maxCycle=j
            bestWeights=np.array(weights)
    weights_array=weights_array.reshape(1000*m,n)
    print("********************************************")
    print(bestWeights)
    print('最佳迭代次数:%d——分类准确率%d%%'%(maxCycle,biggestAccurate*100))
    return bestWeights,maxCycle,weights_array

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
    result=sigmoid(np.dot(dataMat,weight))
    errcount=0
    for i in range(len(labels)):
        if labels[i]==1:
            if result[i]<0.5:
                errcount+=1
        else:
            if (1-result[i])<0.5:
                errcount+=1
    accurte=(len(labels)-errcount)/len(labels)
    return accurte
def pltwightsAndIterationCount(weights1,weights2):
    fig=plt.figure(figsize=(13,7))
    axes1=plt.subplot(321)
    x1=range(0,len(weights1),1)
    y1=weights1[:,0]
    y2 = weights1[:,1]
    y3 = weights1[:,2]

    y4 = weights2[:,0]
    y5 = weights2[:,1]
    y6 = weights2[:,2]

    axes1.plot(x1,y1,color='blue',alpha=0.5)
    axes1.set_ylabel('w0')
    axes1.set_title('随机梯度上升和迭代次数的关系')
    axes2 = plt.subplot(323)
    x2 = range(0, len(weights1))
    axes2.plot(x2, y2, color='blue', alpha=0.5)
    axes2.set_ylabel('w1')

    axes3 = plt.subplot(325)
    x3 = range(0, len(weights1))
    axes3.plot(x3, y3, color='blue', alpha=0.5)
    axes3.set_ylabel('w2')
    axes3.set_xlabel('迭代次数')

    axes4 = plt.subplot(322)
    x4 = range(0, len(weights2)*100,100)
    axes4.plot(x4, y4, color='blue', alpha=0.5)
    axes4.set_ylabel('w0')
    axes4.set_title('批量梯度上升和迭代次数的关系')

    axes5 = plt.subplot(324)
    x5 = range(0, len(weights2)*100,100)
    axes5.plot(x5, y5, color='blue', alpha=0.5)
    axes5.set_ylabel('w1')

    axes6 = plt.subplot(326)
    x6 = range(0, len(weights2)*100,100)
    axes6.plot(x6, y6, color='blue', alpha=0.5)
    axes6.set_ylabel('w2')
    axes6.set_xlabel('迭代次数')

    plt.show()

if __name__ == '__main__':
    dataMat,labels=loadData()
    weight=0
    time1 = time.time()
    weight,maxCycle,weights1=gradScochasticAscent(dataMat,labels)
    time2 = time.time()
    print('Finished in %s seconds' % (time2 - time1))
    plotDataWeights(weight)

    time3=time.time()
    weight, maxCycle, weights2=gradAscent(dataMat,labels)
    time4=time.time()
    print('Finished in %s seconds' % (time4 - time3))
    pltwightsAndIterationCount(weights1,weights2)
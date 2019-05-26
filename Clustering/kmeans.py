#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   kmeans.py
@Time    :   2019/5/26 18:08
@Desc    :

'''
import numpy as np
from random import random
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
def loadDataSet(fileName):
    dataSet=[]
    with open( fileName) as file:
        cons=file.readlines()
    for con in cons:
        curArr=con.strip().split('\t')
        curArr=[float(line)for line in curArr]
        dataSet.append(curArr)
    return dataSet

def calcuDis(VecA,VecB):
    """
        Description:计算两个向量之间的距离
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/26 22:15
    """
    dis= float(np.sqrt((VecA-VecB)*(VecA-VecB).T))
    return dis

def randCen(dataSet,K):
    """
        Description:随机产生K个初始簇中兴
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/26 22:16
    """
    n=np.shape(dataSet)[1]
    centroids=np.mat(np.zeros((K,n)))
    for i in range(K):
        for j in range(n):
            minX = min(dataSet[:, j])
            maxX = max(dataSet[:, j])
            interval = maxX - minX
            r = random()
            centroids[i, j] = minX + interval * r
    return centroids

def plotDataSet(filename):
    dataMat = loadDataSet(filename)  # 加载数据集
    n = len(dataMat)  # 数据个数
    xcord = [];
    ycord = []  # 样本点
    for i in range(n):
        xcord.append(dataMat[i][0]);
        ycord.append(dataMat[i][1])  # 样本点
    fig = plt.figure()
    ax = fig.add_subplot(111)  # 添加subplot
    ax.scatter(xcord, ycord, s=20, c='blue', alpha=.5)  # 绘制样本点
    plt.title('DataSet')  # 绘制title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

def Kmeans(dataSet,K,disMean=calcuDis,createCent=randCen):
    """
        Description:Kmeans实现的具体方法
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/26 22:17
    """
    m=np.shape(dataSet)[0]
    #记录每个点的对应簇索引和最小距离
    clusterAssment=np.zeros((m,2))
    #K个簇
    centroids=randCen(dataSet,K)
    clusterChanged=True
    while clusterChanged :
        clusterChanged=False
        for i in range(m):#遍历每个点
            minDis=float('inf') ;minDisIndex=-1
            for j in range(K):#遍历每个簇
                dis=calcuDis(dataSet[i,:],centroids[j,:])
                if minDis>dis:#得到最小距离的簇
                    minDis=dis
                    minDisIndex=j
            if clusterAssment[i,0]!=minDisIndex:
                clusterChanged=True
                clusterAssment[i,:]=minDisIndex,minDis**2
        for centr in range(K):#更新簇的质心——中心——均值
            indexs=np.nonzero(clusterAssment[:,0]==centr)[0]
            pointsInCluters=dataSet[indexs]
            #axis=0表示按列方向取均值
            centroids[centr,:]=np.mean(pointsInCluters,axis=0)
    return centroids,clusterAssment

def plotClusterResult(dataSet,K,centroids,clusterAssment):
    """
        Description:画聚类的结果
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/26 22:17
    """
    colors=cycle(['r', 'b', 'purple','y','g'])
    plt.figure(figsize=(8,6))
    for i,color in zip(range(K),colors):
        x=dataSet[np.nonzero(clusterAssment[:,0]==i)[0],0].A
        y=dataSet[np.nonzero(clusterAssment[:,0]==i)[0],1].A
        plt.scatter(x,y,c=color,s=30,alpha=0.8)
    plt.scatter(centroids[:,0].A,centroids[:,1].A,c='black',marker='+',s=150)
    plt.title('Clustering result')  # 绘制title
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()


if __name__ == '__main__':
    # plotDataSet('kmeansTestDataSet.txt')
    dataSet=np.mat(loadDataSet('kmeansTestDataSet.txt'))
    K=4
    centroids, clusterAssment=Kmeans(dataSet,K)
    print(centroids)
    print(clusterAssment[:,0])
    plotClusterResult(dataSet,K,centroids,clusterAssment)
    centroids,label,totalS=k_means(dataSet,n_clusters=K)
    print(centroids)
    print(label)
    centroids=np.mat(centroids)
    label=np.mat(label).T
    plotClusterResult(dataSet, K, centroids, label)
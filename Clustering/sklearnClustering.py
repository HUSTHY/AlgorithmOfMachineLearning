#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   sklearnClustering.py
@Time    :   2019/5/27 20:49
@Desc    :   使用sklearn中的算法做聚类分析

'''
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import k_means
from sklearn.cluster import dbscan
from sklearn.cluster import mean_shift
from sklearn.cluster.hierarchical import AgglomerativeClustering #凝聚层次聚类
import time
import numpy as np
def createDataSet():
    x1,y1=datasets.make_circles(n_samples=5000,factor=0.6,noise=0.05)#圈型数据
    x2,y2=datasets.make_blobs(n_samples=1000,n_features=2,centers=[[1.2,1.2]],random_state=9,cluster_std=[[0.1]])#圆型数据
    X=np.concatenate((x1,x2))
    return X


if __name__ == '__main__':
    dataSet=createDataSet()
    plt.scatter(dataSet[:,0],dataSet[:,1],c='black',s=10,alpha=0.8)
    plt.show()
    time1=time.time()
    core_sample,label=dbscan(dataSet,eps=0.08,min_samples=5)#基于密度的带噪声应用空间聚类
    time2=time.time()
    print('dbscan algorithm finshed in %0.4f '%(time2-time1))
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=label, s=10, alpha=0.8)
    plt.title('dbscan clustering')
    plt.show()

    time1 = time.time()
    centoridos, label ,totalS= k_means(dataSet,n_clusters=3)
    time2 = time.time()
    print('k_means algorithm finshed in %0.4f ' % (time2 - time1))
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=label, s=10, alpha=0.8)
    plt.title('k_means clustering')
    plt.show()

    time1 = time.time()
    label= AgglomerativeClustering(n_clusters=3).fit_predict(dataSet)#层次聚类分析
    time2 = time.time()
    print('AgglomerativeClustering algorithm finshed in %0.4f ' % (time2 - time1))
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=label, s=10, alpha=0.8)
    plt.title('hierarchical clustering')
    plt.show()

    time1 = time.time()
    centoridos,label=mean_shift(dataSet)#移动平均聚类
    time2 = time.time()
    print('mean_shift algorithm finshed in %0.4f ' % (time2 - time1))
    plt.scatter(dataSet[:, 0], dataSet[:, 1], c=label, s=10, alpha=0.8)
    plt.title('mean_shift clustering')
    plt.show()
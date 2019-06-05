#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   DataSetDimensionReductions.py
@Time    :   2019/6/4 18:26
@Desc    :

'''
import numpy as np
import pandas as pd
import time
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
from sklearn.decomposition import  FactorAnalysis
from sklearn.decomposition import PCA
from sklearn.decomposition import FastICA
from sklearn.manifold import TSNE
from sklearn.manifold import Isomap
from sklearn import preprocessing

def loadDataSet(fileName):
    dataSet=[]
    dataSet_pd=pd.read_csv(fileName,sep=",")
    # with open(fileName) as file:
    #     cons=file.readlines()
    # for i in range(1,len(cons)):
    #     con=cons[i].strip().split(',')
    #     temp=[]
    #     for j in range(1,len(con)):
    #         temp.append(float(con[j]))
    #     dataSet.append(temp)
    return dataSet_pd,dataSet


def featureDealWithZeroValue(dataSet,percentage):
    time1=time.time()
    dataFea=dataSet.iloc[:,:-1]
    clos=dataFea.columns
    goodCols=[]
    m,n=dataFea.shape
    for i in range(n):
        flag=0
        for j in range(m):
            if dataFea.iloc[j,i]==0:
                flag+=1
        rate=float(flag/n)
        if rate<percentage:
            goodCols.append(clos[i])
    print('0值特征高阈值的有：%d '%len(goodCols),'\n他们是：',goodCols)
    time2 = time.time()
    print('Finshed in %0.4f' % (time2 - time1))
    return dataFea[goodCols]


def featureDealWithLowVar(dataSet):
    """
        Description:方差检验，高于方差阈值的特征留下
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/6/4 21:26
    """
    time1 = time.time()
    datafea=dataSet.iloc[:,:-1]
    datafea=datafea.apply(lambda X:(X-np.min(X))/(np.max(X)-np.min(X)))
    var=datafea.var()
    cols=datafea.columns
    col=[]
    for i in range(len(var)):
        if var[i]>=0:
            col.append(cols[i])
    print('高于阈值的特征个数：%d'%len(col),'\n 它们是:',col)
    time2 = time.time()
    print('Finshed in %0.4f' % (time2 - time1))
    print(datafea[col])
    return datafea[col]

def getColumns(corr,cor=0.9):
    time1 = time.time()
    retCols=[]
    while len(corr)>1:
        m=corr.shape[1]
        cols=corr.columns
        delColums=[]
        retCols.append(cols[0])#首列选取为基准
        delColums.append(cols[0])
        for i in range(1,m):
            if corr.iloc[0][i]>=cor:
                delColums.append(cols[i])
        corr=corr.drop(columns=delColums,index=delColums)#DataFrame中删除多行多列
    retCols.append(corr.columns[0])
    time2 = time.time()
    print('Finshed in %0.4f'%(time2-time1))
    return retCols



def featureDealWithCorr(dataSet):
    """
        Description:特征相关性分析比较
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/6/4 22:42
    """
    time1 = time.time()
    dataFea=dataSet.iloc[:,:-1]#选取数据行和列，得到特征数据
    dataFea=dataFea.apply(lambda X:(X-np.min(X))/(np.max(X)-np.min(X)))#归一化
    corr=dataFea.corr()#得到相关性
    goodCols=getColumns(corr,0.9)#相关性分析
    time2 = time.time()
    print('Finshed in %0.4f' % (time2 - time1))
    print(dataFea[goodCols])
    return dataFea[goodCols]

def featureDealWithFeaImportanceInRandomForest(dataSet):
    """
        Description:使用RandomForest中的feature_importances_来确定特征的重要性
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/6/5 13:13
    """
    time1 = time.time()
    dataFea=dataSet.iloc[:,:-1]
    dataFea=dataFea.apply(lambda X:(X-np.min(X))/(np.max(X)-np.min(X)))
    label=dataSet.iloc[:,-1]
    #随机森林
    rf=RandomForestRegressor(n_estimators=80,max_depth=10,oob_score=True,random_state=10)
    dataFea=dataFea.fillna(0)
    rf.fit(dataFea,label)


    features=dataFea.columns
    feaImportacnes=rf.feature_importances_
    time2 = time.time()
    print('Finshed in %0.4f' % (time2 - time1))

    indicates=np.argsort(-feaImportacnes)[0:15]#降序得到索引值
    print(feaImportacnes[indicates])

    plt.figure(figsize=(13,7))
    plt.barh(range(len(indicates)),feaImportacnes[indicates],color='r',align='center')
    plt.yticks(range(len(indicates)),[features[indicate] for indicate in indicates])
    plt.show()


def featureDealWithFactorAnalysis(dataSet):
    """
        Description: 因子分析法——FactorAnalysis
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/6/5 14:28
    """
    time1 = time.time()
    dataFea=dataSet.iloc[:,:-1]
    dataFea=dataFea.fillna(0)
    cols=dataFea.columns

    #标准化
    dataFea_mean=dataFea.mean()
    dataFea_std=dataFea.std()
    goodCol=[]
    #把方差太小的过滤出来
    for i in range(len(dataFea_std)):
        if dataFea_std[i]>0.001:
            goodCol.append(cols[i])
    dataFea_mean=dataFea_mean[goodCol]
    dataFea_std=dataFea_std[goodCol]
    dataFea_Copy= (dataFea[goodCol]-dataFea_mean)/dataFea_std
    values=dataFea_Copy.values
    #潜在因子分析——差不多已经降维完成了
    FA=FactorAnalysis(n_components=10).fit_transform(values)

    #FA归一化
    min_max_scaler=preprocessing.MinMaxScaler()
    FA=min_max_scaler.fit_transform(FA)
    time2 = time.time()
    print('Finshed in %0.4f' % (time2 - time1))
    print(np.shape(FA))
    for i in range(9):
        plt.scatter(FA[:,i],FA[:,i+1],s=20,alpha=0.7)
    plt.scatter(FA[:,9],FA[:,0],s=20,alpha=0.7)
    plt.show()
    return FA

def featureDealWithPCA(dataSet):
    """
        Description:主成分分析
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/6/5 15:45
    """
    time1 = time.time()
    dataFea = dataSet.iloc[:, :-1]
    dataFea = dataFea.fillna(0)
    cols = dataFea.columns

    # 标准化
    dataFea_mean = dataFea.mean()
    dataFea_std = dataFea.std()
    goodCol = []
    # 把方差太小的过滤出来
    for i in range(len(dataFea_std)):
        if dataFea_std[i] > 0.001:
            goodCol.append(cols[i])
    dataFea_mean = dataFea_mean[goodCol]
    dataFea_std = dataFea_std[goodCol]
    dataFea_Copy = (dataFea[goodCol] - dataFea_mean) / dataFea_std
    values = dataFea_Copy.values


    totalR = 0
    pca = PCA(n_components=0.80)#映射新的坐标系后，数据保留原有的80%的信息
    pca_res = pca.fit_transform(values)  # 降维后的数据，没有归一化
    time2 = time.time()
    print('Finshed in %0.4f' % (time2 - time1))
    explainRatios = pca.explained_variance_ratio_  # 每个特征的解释性
    for i in explainRatios:#累计特征的解释性
        totalR += i
    print(totalR)
    k=len(explainRatios)
    print(k)
    plt.figure(figsize=(20,8))
    plt.bar(range(k),explainRatios,color='b',label='single interpretation variance')
    plt.plot(range(k),np.cumsum(explainRatios),c='r',label='Cumlative explain variance')
    plt.legend()
    plt.show()
    return pca_res

def featureDealWithICA(dataSet):
    time1 = time.time()
    dataFea = dataSet.iloc[:, :-1]
    dataFea = dataFea.fillna(0)
    cols = dataFea.columns

    # 标准化
    dataFea_mean = dataFea.mean()
    dataFea_std = dataFea.std()
    goodCol = []
    # 把方差太小的过滤出来
    for i in range(len(dataFea_std)):
        if dataFea_std[i] > 0.001:
            goodCol.append(cols[i])
    dataFea_mean = dataFea_mean[goodCol]
    dataFea_std = dataFea_std[goodCol]
    dataFea_Copy = (dataFea[goodCol] - dataFea_mean) / dataFea_std
    values = dataFea_Copy.values

    # 独立分量分析（ICA）
    n_component=3
    ICA = FastICA(n_components=n_component, random_state=12)
    X = ICA.fit_transform(values)
    time2 = time.time()
    print('Finshed in %0.4f' % (time2 - time1))

    # 归一化

    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # 绘制图像，观察成分独立情况
    plt.figure(figsize=(12, 5))
    plt.title('Factor Analysis Components')
    for i in range(n_component-1):
        plt.scatter(X[:,i],X[:,i+1])
    plt.scatter(X[:, n_component-1], X[:,0])
    plt.show()


def featureDealWithTSNE(dataSet):
    time1 = time.time()
    dataFea = dataSet.iloc[:, :-1]
    dataFea = dataFea.fillna(0)
    cols = dataFea.columns

    # 标准化
    dataFea_mean = dataFea.mean()
    dataFea_std = dataFea.std()
    goodCol = []
    # 把方差太小的过滤出来
    for i in range(len(dataFea_std)):
        if dataFea_std[i] > 0.001:
            goodCol.append(cols[i])
    dataFea_mean = dataFea_mean[goodCol]
    dataFea_std = dataFea_std[goodCol]
    dataFea_Copy = (dataFea[goodCol] - dataFea_mean) / dataFea_std
    values = dataFea_Copy.values

    # TSNE分析
    n_component = 3
    tsne = TSNE(n_components=n_component, random_state=12)
    X = tsne.fit_transform(values)
    time2 = time.time()
    print('Finshed in %0.4f' % (time2 - time1))

    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # 绘制图像
    plt.figure(figsize=(12, 5))
    plt.title('Factor Analysis Components')
    for i in range(n_component - 1):
        plt.scatter(X[:, i], X[:, i + 1])
    plt.scatter(X[:, n_component - 1], X[:, 0])
    plt.show()



def featureDealWithISOMAP(dataSet):
    time1 = time.time()
    dataFea = dataSet.iloc[:, :-1]
    dataFea = dataFea.fillna(0)
    cols = dataFea.columns

    # 标准化
    dataFea_mean = dataFea.mean()
    dataFea_std = dataFea.std()
    goodCol = []
    # 把方差太小的过滤出来
    for i in range(len(dataFea_std)):
        if dataFea_std[i] > 0.001:
            goodCol.append(cols[i])
    dataFea_mean = dataFea_mean[goodCol]
    dataFea_std = dataFea_std[goodCol]
    dataFea_Copy = (dataFea[goodCol] - dataFea_mean) / dataFea_std
    values = dataFea_Copy.values

    # TSNE分析
    n_component = 3
    isomap = Isomap(n_components=n_component,n_neighbors=5,n_jobs=-1)
    X = isomap.fit_transform(values)
    time2 = time.time()
    print('Finshed in %0.4f' % (time2 - time1))

    # 归一化
    min_max_scaler = preprocessing.MinMaxScaler()
    X = min_max_scaler.fit_transform(X)

    # 绘制图像
    plt.figure(figsize=(12, 5))
    plt.title('Factor Analysis Components')
    for i in range(n_component - 1):
        plt.scatter(X[:, i], X[:, i + 1])
    plt.scatter(X[:, n_component - 1], X[:, 0])
    plt.show()

if __name__ == '__main__':
    dataSet_pd,dataSet=loadDataSet('slice_localization_data.csv')
    featureDealWithLowVar(dataSet_pd)
    featureDealWithCorr(dataSet_pd)
    featureDealWithFeaImportanceInRandomForest(dataSet_pd)
    featureDealWithFactorAnalysis(dataSet_pd)
    featureDealWithPCA(dataSet_pd)
    featureDealWithICA(dataSet_pd)
    # featureDealWithISOMAP(dataSet_pd)
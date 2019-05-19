#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   MultClassROCAndAUC.py
@Time    :   2019/5/19 15:05
@Desc    :   调用sklearn相关函数实现多分类的ROC和AUC曲线的绘制

'''

from sklearn import datasets
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from itertools import cycle
import numpy as np
from scipy import interp
from sklearn.metrics import roc_auc_score
def showMultiClassROCAndAUC():
    iris=datasets.load_iris()
    trainDataSet=iris.data
    trainLabel=iris.target
    #测试集按照0.3的比例选取随机选取，然后random_state保证程序每次选取的随机数都是一样的
    x_train,x_test,y_train,y_test=train_test_split(trainDataSet,trainLabel,test_size=0.5,random_state=1)
    #二值化
    y_test=label_binarize(y_test,classes=[0,1,2])
    svm=SVC(C=1.0,kernel='linear',degree=2,probability=True,gamma=0.1)
    clf=OneVsRestClassifier(svm)
    clf.fit(x_train, y_train)
    y_score=clf.decision_function(x_test)
    showMacroROCAndAUC(y_test,y_score)
    showMicroROCAndAUC(y_test,y_score)


def showMacroROCAndAUC(y_test,y_score):
    """
        Description:把预测强度矩阵和标签矩阵的每一列做基准画ROC曲线，可以得到n条ROC曲线
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/19 17:54
    """
    m,n=y_test.shape
    FPRs=[]  ;TPRs=[]
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i,color in zip(range(n),colors):
        FPR,TPR,threshodls=roc_curve(y_test[:,i],y_score[:,i])
        FPRs.append(FPR)
        TPRs.append(TPR)
        AUC=auc(FPR,TPR)
        plt.plot(FPR,TPR,c=color,label='AUC is %4f'%AUC)
        plt.xlabel('TPR')
        plt.ylabel('FPR')
        plt.title('Macro ROC and AUC ')
    #求平均的TPR和FPR
    all_fpr = np.unique(np.concatenate([fpr for fpr in FPRs]))#也就是平均后了的FPR
    mean_tpr=np.zeros_like(all_fpr)
    for FPR,TPR in zip(FPRs,TPRs):
        #需要进行插值计算
        mean_tpr+=interp(all_fpr,FPR,TPR)
    mean_tpr/=n
    AUC_mean=auc(all_fpr,mean_tpr)
    plt.plot(all_fpr, mean_tpr, c='r', label='Macro ROC AUC is %4f' % AUC_mean)
    plt.plot([0,1],[0,1],c='black',linestyle='--',lw=2)
    print('直接调用roc_auc_score函数macro得到auc的值：%4f'%roc_auc_score(y_test,y_score,average='macro'))

def showMicroROCAndAUC(y_test,y_score):
    """
        Description:把预测强度矩阵和标签矩阵降维y_score.ravel()转置后，可以画出一条整体的ROC曲线
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/19 17:56
    """
    FPR, TPR, threshodls = roc_curve(y_test.ravel(),y_score.ravel())
    AUC=auc(FPR,TPR)
    plt.plot(FPR,TPR,c='purple',label='Micro ROC and AUC is:% 4f'%AUC)
    plt.legend()
    plt.show()
    print('直接调用roc_auc_score函数micro得到auc的值：%4f' % roc_auc_score(y_test, y_score, average='micro'))

if __name__ == '__main__':
    showMultiClassROCAndAUC()
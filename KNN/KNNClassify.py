import numpy as np
import  operator as op
import time
def creatTrainDataset():
    group=np.mat('1,101;5,89;108,5;115,8;22,99')
    labels=['爱情片','爱情片','动作片','动作片','爱情片']
    return group,labels
#

'''KNN算法分类器实现'''
"""
函数说明：KNN算法分类器
parameters：
    InputNewData-用于分类的数据（测试集），结构形式可以是数组可以是矩阵
    trainDataset—用于训练的数据（训练集）
    labels-分类标签
    K-KNN算法参数，距离最小的K个点值
    rule—分类规则，0表示多数表决制rule=1表示距离加权
    为了代码实现方便，最好使用array来操作

Modify:
    2019-04-17
"""
def KNNClassify(InputNewData,trainDataset,labels,K,rule):
    #shape获取行的维度
    newDataDim=InputNewData.shape[0]
    OutclassItems=[]
    for j in range(newDataDim):
        InputNewDataRow=InputNewData[j]
        trainDataDimension=trainDataset.shape[0]
        #tile（）复制矩阵trainDataDimension行，1列
        InputNewDataSet=np.tile(InputNewDataRow,(trainDataDimension,1))
        diffMat=InputNewDataSet-trainDataset
        #矩阵点乘，相同下标的元素相乘
        sqDiffMat=np.multiply(diffMat,diffMat)
        #矩阵行方向上元素求和，得到一个新的矩阵——这是一个行向量
        sqDistanceList=sqDiffMat.sum(axis=1) #这是个行向量
        #矩阵元素开方列，得到真实距离
        distanceList=np.sqrt(sqDistanceList)
        #距离排序后得到一个
        sortedDistanceList=distanceList.reshape(1,trainDataDimension).argsort().reshape(trainDataDimension,1)#矩阵中argsort()函数只能是列向量才有用，数组没要求

        #记录分类类别次数的字典
        classItems={}

        for i in range(K):
            #取出前K个元素的类别
            voteLable=labels[int(sortedDistanceList[i])]#sortedDistanceList[i]这里必须是行向量，列向量的话只又一个值，显然错误
            #统计类别的次数
            if rule==0:
                classItems[voteLable]=classItems.get(voteLable,0)+1
            elif rule==1:
                classItems[voteLable]=classItems.get(voteLable,0)+1/(sortedDistanceList[i]**2)
            else:
                print('分类规则错误！请修改参数！')
                break
        #降序排序字典itemgetter(1)按照字典的value排序
        sortedClassItems=sorted(classItems.items(),key=op.itemgetter(1),reverse=True)
        # print(sortedClassItems)
        OutclassItems.append(sortedClassItems[0][0])
    return OutclassItems


'''运行main函数的时候就打开注释'''
# if __name__ == '__main__':
#     group,labels=creatTrainDataset()
#     test=np.mat('101,20;35,99')
#     time0=time.time()
#     t=KNNClassify(test,group,labels,2,0)
#     time1=time.time()
#     print('耗时：',time1-time0)
#     print(t)
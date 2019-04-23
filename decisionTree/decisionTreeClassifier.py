from math import log
import operator
import pickle

def createDataSet():
    """
      Description:
      Params:
            No such property: code for class: Script1
      Return:
            dataSet——特征数据
            labels——分类标签
      Author:
            HY
      Modify:
            2019/4/22 15:03
    """
    dataSet = [[0, 0, 0, 0, 'no'],         #数据集
               [0, 0, 0, 1, 'no'],
               [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'],
               [0, 0, 0, 0, 'no'],
               [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'],
               [1, 1, 1, 1, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'],
               [2, 0, 1, 2, 'yes'],
               [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'],
               [2, 1, 0, 2, 'yes'],
               [2, 0, 0, 0, 'no']]
    labels = ['年龄', '有工作','有自己的房子', '信贷情况']             #分类属性
    return dataSet, labels                #返回数据集和分类属性

def calShannonEnt(dataSet):
    """
      Description:计算香农熵
      Params:
            dataSet——数据集
      Return:
            shannonent——训练集的香农熵
      Author:
            HY
      Modify:
            2019/4/22 15:37
    """
    NumVec=len(dataSet)
    shannonent=0
    labelsDic={}
    for featureVector in dataSet:
        currentLabel=featureVector[-1]
        if currentLabel not in labelsDic.keys():
            labelsDic[currentLabel]=0
        labelsDic[currentLabel]+=1
    prob=0
    for key in labelsDic.values():
        prob=float(key/NumVec)
        shannonent-=prob*log(prob,2)
    return shannonent

def splitDataSet(dataSet,feature,featureVal):
    """
      Description:把数据集按照选定的特征，根据特征值的不同分组，同时把该特征对应的数据列去掉
      Params:
            dataSet——数据集
            feature——特征(特征索引）
            featureVal——特征值
      Return:
            retDataset——返回的只数据集
      Author:
            HY
      Modify:
            2019/4/22 16:11
    """
    retDataset=[]
    for v in dataSet:
        if v[feature]==featureVal:
            #v的切片，v1、v2把选定的那一列特征去掉了
            v1=v[:feature]
            v2=v[feature+1:]
            # retVec=v1.extend(v2)
            # retDataset.append(retVec)
            v1.extend(v2)
            retDataset.append(v1)
    return retDataset

def calInfoGainAndChooseBsetFeature(dataSet):

    """
      Description:计算每个特征的信息增益
      Params:
            dataSet——数据集
      Return:
            bestFeatureIndex——最佳的特征
      Author:
            HY
      Modify:
            2019/4/22 16:42
    """

    wholeEntropy=calShannonEnt(dataSet)
    biggestGain=0
    biggestGain=-1
    featureNum=len(dataSet[0])-1
    for f in range(featureNum):
        featurlist=[fea[f] for fea in dataSet]
        uniqueFeaturelist=set(featurlist)
        newEntropy=0
        for value in uniqueFeaturelist:
            subDataset=splitDataSet(dataSet,f,value)#按照选定的特征，根据特征值的不同分组
            prob=float(len(subDataset)/len(dataSet))
            newEntropy+=prob*calShannonEnt(subDataset)#条件熵的计算
        gainInfo=wholeEntropy-newEntropy#信息增益
        if gainInfo>=biggestGain:
            biggestGain=gainInfo
            bestFeatureIndex=f
    return bestFeatureIndex


def majrotyCnt(classList):

    """
      Description:统计分类列表中种类最多的那一类
      Params:
            classList——分类标签列表
      Return:
            sortedClassDic[0][0]——返回分类次数最多的那一类
      Author:
            HY
      Modify:
            2019/4/22 23:22
    """

    ClassDic={}
    for e in classList:
        if e not in ClassDic.keys():
            ClassDic[e]=0
        ClassDic[e]+=1
    sortedClassDic=sorted(ClassDic.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassDic[0][0]

def creatDecisionTree(dataSet,labels,bestFeatureLabels):

    """
      Description:使用递归构建一颗决策树
      Params:
            dataSet—— 数据集
            labels——标签列表
            bestFeatureLabels——最优特征对应的标签
      Return:
            No such property: code for class: Script1
      Author:
            HY
      Modify:
            2019/4/22 23:24
    """

    classList=[e[-1] for e in dataSet]
    # 两个if语句就是递归的终止条件
    if classList.count(classList[0])==len(classList):
        return classList[0]
    if len(dataSet[0])==1 or len(labels)==0:
        return majrotyCnt(classList)
    bestFeatureIndex=calInfoGainAndChooseBsetFeature(dataSet)#得到最优特征的索引
    bestFeatureLabel=labels[bestFeatureIndex]
    bestFeatureLabels.append(bestFeatureLabel)#记录每次对应的最优特征标签
    decisionTree={bestFeatureLabel:{}}        #树的结构 字典保存
    bestFeatureValues=[e[bestFeatureIndex] for e in dataSet]
    uniqueBestFeatureValues=set(bestFeatureValues)
    del(labels[bestFeatureIndex])
    for v in uniqueBestFeatureValues:
        newDataSet=splitDataSet(dataSet,bestFeatureIndex,v)#按照最优特征的每一项值来划分数据集
        decisionTree[bestFeatureLabel][v]=creatDecisionTree(newDataSet,labels,bestFeatureLabels)#这里就是递归算法
    return decisionTree

def decisionTreeClassifier(decisionTree,featuresLabels,testVec):

    """
      Description:使用决策树对测试集进行分类；就是把决策树对应的规则路径和测试集做对比，就能得到最终结果。需要用到递归的思想
      Params:
            decisionTree——构建好的决策树
            featuresLabels——最优的特征标签列表
            testVec——对应于最优的特征标签列表来赋值
      Return:
            classLabel——返回分类的结果
      Author:
            HY
      Modify:
            2019/4/22 23:29
    """

    firstNode=next(iter(decisionTree))#获取树的跟节点
    firstDic=decisionTree[firstNode]#获取节点对应的值
    featureIndex=featuresLabels.index(firstNode)
    for key in firstDic.keys():
        if testVec[featureIndex]==key:
            if type(firstDic[key]).__name__=='dict':#如果节点值的数据类型是字典，就进行递归
                classLabel=decisionTreeClassifier(firstDic[key],featuresLabels,testVec)
            else:#如果是其他类型就直接返回分类结果
                classLabel=firstDic[key]
    return classLabel

def storeDecisionTree(decisionTree,fileName):
    #w写str;wb写bytes
    file =open(fileName,'wb')
    pickle.dump(decisionTree,file)

def grapDecisionTree(fileName):
    with open(fileName,'rb') as file:
        return pickle.load(file)



if __name__ == '__main__':
    dataSet,labels=createDataSet()
    bestFeatureLabels=[]
    decisionTree=creatDecisionTree(dataSet,labels,bestFeatureLabels)
    testVec=[0,1]#没有自己的房子；有工作
    classLabel=decisionTreeClassifier(decisionTree,bestFeatureLabels,testVec)#bestFeatureLabels在creatDecisionTree()函数中已经改变——和JAVA不一样
    if classLabel=='no':
        print('不给于贷款')
    else:
        print('给与放贷')
    storeDecisionTree(decisionTree,'storeDecisionTree')
    print(grapDecisionTree('storeDecisionTree'))




import decisionTree.decisionTreeClassifier as OWNDecTreeClassifier #自己写的决策树分类器
from sklearn import tree
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def useOwnDecisionTreeForClassify(dataSet,labels,testVect):

    """
      Description:
      Params:
            dataSet——数据集，包含了分类列，便于计算信息熵、增益等
            labels——标签集，特征名的list
            testVect——用来预测的数据
      Return:
            resultClassify——返回分类结果
      Author:
            HY
      Modify:
            2019/4/24 15:24
    """

    bestFeatureLabels=[]
    ownDecTree=OWNDecTreeClassifier.creatDecisionTree(dataSet,labels,bestFeatureLabels)
    resultClassify=OWNDecTreeClassifier.decisionTreeClassifier(ownDecTree,bestFeatureLabels,testVect)
    return resultClassify

def getLensesDataset(fileName):
    with open(fileName,'r') as file:
        lenses=[elem.strip().split('\t')for elem in file.readlines()]
        return lenses


def useSKLearn_DecisionTreeForClassify(X,Y,testVect):

    """
      Description:
      Params:
            X——训练集的自变量
            Y——训练集的因变量
            testVect——要预测的数据特征向量
      Return:
            No such property: code for class: Script1
      Author:
            HY
      Modify:
            2019/4/23 22:14
    """

    dTree=tree.DecisionTreeClassifier(criterion='entropy')#选用gini还是信息增量entropy
    dTree.fit(X,Y)#训练树
    result=dTree.predict(testVect)
    print(result)

def preprocessingData(dataSet,labels):

    """
      Description: 数据预处理
      sklearn中fit函数不支持字符串，所以要把数据换成float型
      使用pandas和labelEncoder序列化数据
      先把原始数据的特征和特征值用字典保存———然后字典生成pandas格式数据表
      ————最后吧pandas数据进行序列化
      labelEncoder.fit_transform把字符串转化为增量值
      Params:
            dataSet——数据集
            labels——标签集
      Return:
            lensesPd——返回序列化后的数据集（只有自变量）
            targetList——目标集（因变量）
      Author:
            HY
      Modify:
            2019/4/23 20:28
    """

    targetList=[]
    lenses_dic={}
    for elem in dataSet:
        targetList.append(elem[-1])
    for label in labels:
        lensesList=[]
        for elem in dataSet:
            lensesList.append(elem[labels.index(label)])
        lenses_dic[label]=lensesList
    # print(lenses_dic)
    lensesPd=pd.DataFrame(lenses_dic)#生成pandas格式的数据类似一张Excel表
    labelEnc=LabelEncoder()
    for col in lensesPd.iteritems():#iteritems()按列遍历iterrows()和itertuples()按行遍历
        # col[0]列名——age,col[1]列的值——是一个列表，labelEnc.fit_transform(col[1])——这样才可以的
        lensesPd[col[0]]=labelEnc.fit_transform(col[1])#把每一列进行序列化
    print(lensesPd)#对照这个序列化编码后的才能构造对应的测试集
    return lensesPd,targetList






if __name__ == '__main__':
    dataSet=getLensesDataset('lenses.txt')
    dataSet1=getLensesDataset('lenses.txt')
    labels=['age','prescript症状','astigmatio散光','tearRate眼泪数量']
    labels1=['age','prescript症状','astigmatio散光','tearRate眼泪数量']
    testVec=['normal','yes','hyper','pre']
    print(useOwnDecisionTreeForClassify(dataSet,labels,testVec))
    lensespd,targetList=preprocessingData(dataSet1,labels1)
    testVec1=[[0,0,1,0]]#不是按照最优的特征来的，而是按照标签列序列化映射关系而来的
    result=useSKLearn_DecisionTreeForClassify(lensespd.values.tolist(),targetList,testVec1)
    # useSKLearn_DecisionTreeForClassify(dataSet,labels1,testVec)

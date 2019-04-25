import numpy as np
import scipy.stats as ss
def getDataSet(fileName):
    with open(fileName,encoding='utf-8')as file:
        dataSet=[ele.strip().split('\t') for ele in file.readlines()]
        return dataSet

def naiveBayesianClassifier(dataSet,testVec,type):

    """
      Description:
      Params:
            dataSet——训练数据
            testVec——用来分类的数据集
            type——数据集离散还是连续,0离散的，1是连续的
      Return:
            BiggestPostPro——最大的后验概率
            result——分类结果
      Author:
            HY
      Modify:
            2019/4/25 21:18
    """
    if type==0:
        return naiveBayesianClassifierForDiscreteData(dataSet,testVec)
    else:
        return naiveBayesianClassifierForContinuousData(dataSet,testVec)

def naiveBayesianClassifierForDiscreteData(dataSet,testVec):
    targetList=[]
    for ele in dataSet:#目标集
        targetList.append(ele[-1])
    targetDic={}#统计目标集分类情况
    for ele in targetList:
        if ele not in targetDic.keys():
            targetDic[ele]=0
        targetDic[ele]+=1
    BiggestPostPro=0
    result=None
    for key,val in targetDic.items():#计算后验概率对每一个分类都要计算
        PostPro=float(int(val)/len(targetList))
        subDataset=[]
        for ele in dataSet:#进行分类
            if ele[-1]==key:
                subDataset.append(ele)
        for tes in testVec:#特征之间相互独立，计算指定分类情况和特征的下，某一特征值的概率
            list=[]
            for s in subDataset:#统计相同特征值的数目
                if tes==s[testVec.index(tes)]:
                    list.append(s)
            PostPro*=float(len(list)/len(subDataset))
        if BiggestPostPro<PostPro:
            BiggestPostPro=PostPro
            result=key
    return BiggestPostPro,result

def naiveBayesianClassifierForContinuousData(dataSet,testVec):
    BiggestPsotProb=0
    targetList=[]
    targetDic={}
    result=None
    for ele in dataSet:
        targetList.append(ele[-1])
    for ele in targetList:
        if ele not in targetDic:
            targetDic[ele]=0
        targetDic[ele]+=1
    for key,val in targetDic.items():
        prob=float(val/len(targetList))
        subDataSet=[]
        for ele in dataSet:
            if ele[-1]==key:
                subDataSet.append(ele)
        for t in testVec:
            index=testVec.index(t)
            coluList=[]
            for ele in subDataSet:
                coluList.append(float(ele[index]))
            #计算方法和期望用到numpy，计算高斯分布用到scipy.stats.norm(excep,std).pdf(t)这个函数链式很关键
            excep=np.mean(coluList)
            std=np.std(coluList)
            prob*=ss.norm(excep,std).pdf(t)
        if BiggestPsotProb<prob:
            BiggestPsotProb=prob
            result=key
    return  BiggestPsotProb,result

if __name__ == '__main__':
    dataSet=getDataSet('girlMarryToBoy.txt')
    testVec=['帅','不好','高','不上进']
    print(naiveBayesianClassifier(dataSet,testVec,0))
    dataSet1=getDataSet('SexClassificationDataSet.txt')
    testVec1=[6,130,8]
    print(naiveBayesianClassifier(dataSet1,testVec1,1))



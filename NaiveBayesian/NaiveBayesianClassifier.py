import numpy as np
import scipy.stats as ss
def getDataSet(fileName):
    with open(fileName,encoding='utf-8')as file:
        dataSet=[ele.strip().split('\t') for ele in file.readlines()]
        return dataSet

def naiveBayesianClassifier(dataSet,testVec):

    """
      Description:计算中涉及到离散和连续型数据都做了处理——这里连续型只考虑使用概率分布的方法来做，而不是考虑离散化数据
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
                if not ele[index].isalpha():
                    coluList.append(ele[index])
                else:
                    if t==ele[index]:
                        coluList.append(ele[index])
            #计算方法和期望用到numpy，计算高斯分布用到scipy.stats.norm(excep,std).pdf(t)这个函数链式很关键
            if not coluList[0].isalpha():#判定不是字母
                l=list(map(float,coluList))
                excep=np.mean(l)
                # 这里要用到极大似然估计，方差是/n 而不是/n-1
                var=np.var(l)*len(coluList)/(len(coluList)-1)
                #标准差
                std=pow(var,0.5)
                #参数为均值和标准差
                prob*=ss.norm(excep,std).pdf(t)
            else:#计算离散变量
                prob*=float(len(coluList)/len(subDataSet))

        if BiggestPsotProb<prob:
            BiggestPsotProb=prob
            result=key
    return BiggestPsotProb,result

if __name__ == '__main__':
    dataSet=getDataSet('girlMarryToBoy.txt')
    testVec=['不帅','不好','矮','不上进']
    print(naiveBayesianClassifier(dataSet,testVec))
    dataSet1=getDataSet('SexClassificationDataSet.txt')
    testVec1=[6,130,8]
    print(naiveBayesianClassifier(dataSet1,testVec1))





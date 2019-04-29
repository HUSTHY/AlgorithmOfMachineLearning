import re
import numpy as np
import random

def textPares(bigString):

    """
      Description:接受一个打字符串并且将其解析为字符串列表
      Params:
            No such property: code for class: Script1
      Return:
            No such property: code for class: Script1
      Author:
            HY
      Modify:
            2019/4/27 15:25
    """
    listOfTokens=re.split(r'\W+',bigString)#\W 正则表达式表示非字母和数字
    return [lt.lower() for lt in listOfTokens if len(lt)>2]

def createVocabList(dataSet):

    """
      Description:创建词汇表——把样本中的词条去重得到列表
      Params:
            No such property: code for class: Script1
      Return:
            No such property: code for class: Script1
      Author:
            HY
      Modify:
            2019/4/27 15:30
    """

    vocabSet=set([])
    for ele in dataSet:
      vocabSet=vocabSet | set(ele)
    return vocabSet

def setOfWordsTwoValVec(vocabList,inputSet):

    """
      Description:根据词汇表，把输入的数据向量化——词集模型
      Params:
            No such property: code for class: Script1
      Return:
            No such property: code for class: Script1
      Author:
            HY
      Modify:
            2019/4/27 15:37
    """

    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print('the word: %s is not in the Vocabulary.'% word)
    return  returnVec

def setBagOfWordsTwoValVec(vocabList,inputSet):

    """
      Description:词袋模型
      Params:
            No such property: code for class: Script1
      Return:
            No such property: code for class: Script1
      Author:
            HY
      Modify:
            2019/4/27 19:18
    """

    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]+=1
        else:
            print('the word: %s is not in the Vocabulary.'% word)
    return  returnVec

def trainNBS(trainMatrix,trainCategory):
    
    """
      Description:把setOfWordsTwoValVec()返回的returnVec构成的矩阵和trainCategory进行训练，得出先验概率、和条件概率组
      trainMatrix这个矩阵是由文档经过词汇表向量化得来的，因此每一行的维度是一样的
      使用的是多项式模型,同时要使用laplcae平滑处理
      下溢出处理——取对数——（两个较小的数越乘越小）
      Params:
            No such property: code for class: Script1
      Return: 
            No such property: code for class: Script1
      Author: 
            HY
      Modify: 
            2019/4/27 16:26
    """
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    ClassSpamtotalWords=0
    allWords=0
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            ClassSpamtotalWords+=sum(trainMatrix[i])
        allWords+=sum(trainMatrix[i])
    pAbusive=float(ClassSpamtotalWords/allWords)
    p0Num=np.ones(numWords)
    p1Num=np.ones(numWords)
    p0Denom=2
    p1Denom=2
    for i in range(numTrainDocs):
        if trainCategory[i]==1:
            p1Num+=trainMatrix[i]
            p1Denom+=sum(trainMatrix[i])
        else:
            p0Denom+=sum(trainMatrix[i])
            p0Num+=trainMatrix[i]
    p1Vec=np.log(p1Num/p1Denom)
    p0Vec=np.log(p0Num/p0Denom)
    return pAbusive,p1Vec,p0Vec


def classifyNB(vecTwoClassify,p0Vec,p1Vec,pSpam):
    p1=sum(vecTwoClassify*p1Vec)+np.log(pSpam)
    p0=sum(vecTwoClassify*p0Vec)+np.log(pSpam)
    # if p1>p0:
    #     return 1
    # else:
    #     return 0
    return p1>p0

def spamMailTest():

    """
      Description: 使用的是多项式模型,同时要使用laplcae平滑处理，要使用词袋模型来计数
      Params:
            No such property: code for class: Script1
      Return:
            No such property: code for class: Script1
      Author:
            HY
      Modify:
            2019/4/27 21:48
    """

    docList=[];classList=[];fullList=[]
    for i in range(1,26):
        wordList=textPares(open('spam/%d.txt'% i).read())
        docList.append(wordList)
        classList.append(1)
        fullList.append(wordList)
        wordList=textPares(open('ham/%d.txt'% i).read())
        docList.append(wordList)
        classList.append(0)
        fullList.append(wordList)
    #词汇表
    vocabList=list(createVocabList(docList))
    trainingSet=list(range(50));testSet=[]
    for i in range(10):#随机选取10个文档作为测试集——与分类的文档
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[];trainClasses=[]
    for docIndex in trainingSet:
        #使用的是多项式模型,同时要使用laplcae平滑处理，要使用词袋模型来计数，而不适用词集模型
        trainMat.append(setBagOfWordsTwoValVec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    pSpam,p1v,p0v=trainNBS(trainMat,trainClasses)
    errorCount=0
    for docIndex in testSet:
        wordVector=setOfWordsTwoValVec(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0v,p1v,pSpam)!= classList[docIndex]:
            errorCount+=1
            print('分类错误的测试集：',docList[docIndex])
    print('总计错误分类为：',errorCount)
if __name__ == '__main__':
    spamMailTest()

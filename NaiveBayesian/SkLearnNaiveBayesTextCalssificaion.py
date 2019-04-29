import os
import jieba
from sklearn import naive_bayes as nb
import copy
import time
import matplotlib.pyplot as plt

"""
使用sklearn库的朴素贝叶斯来进行文本分类
思路：
    1、收集到的数据进行整理，放置在类名的文件夹下面
    2、读取文件夹下面的文件，获得训练集数据
    3、训练集数据进行——特征词选取——词汇表的构建、文档向量化
    4、调用贝叶斯分类器进行数据的训练
    5、分类预测
"""

def TextProcessing(folder_path):

    """
      Description:从文件夹中获取文件中的数据
      Params:
            folder_path——文件路径
      Return:
            all_words_list——总单词列表去掉重复的
            calssList——分类列表
            dataList——数据列表
      Author:
            HY
      Modify:
            2019/4/29 21:34
    """

    folder_list=os.listdir(folder_path)
    dataList=[]
    calssList=[]

    for folder in folder_list:
        sub_folder_path=os.path.join(folder_path,folder)#根据子文件夹生成新的路径
        fileNames=os.listdir(sub_folder_path)#获取了文件夹下所有的文件名列表
        for fileName in fileNames:
            with open(os.path.join(sub_folder_path,fileName),'r') as file:
                content=file.read()
            word_cut=jieba.cut(content,cut_all=True)#全模式分词
            word_list=list(word_cut)
            dataList.append(word_list)
            calssList.append(folder)
    all_words_dic={}
    for dl in  dataList:
        for word in dl:
            if word in all_words_dic.keys():
                all_words_dic[word]+=1
            else:
                all_words_dic[word]=1
    #排序——按照val降序排列
    sorted_all_words_list=sorted(all_words_dic.items(),key=lambda d:d[1],reverse=True)
    #字典的键和值分开解压
    all_words_list,all_words_num=zip(*sorted_all_words_list)
    return all_words_list,calssList,dataList

def createFeatureVocabulary(all_words_list,deleteN,stopWords_set=set()):
    featureVocabulary=[]
    demension=1
    for i in range(deleteN,len(all_words_list),1):
        if demension>1000:
            break
        if not all_words_list[i].isdigit() and all_words_list[i] not in stopWords_set and 1<len(all_words_list[i])<5:
            featureVocabulary.append(all_words_list[i])
            demension+=1
    return  featureVocabulary

def getStopWordsList(fileName):
    stopWords=[]
    with open(fileName,'rb') as file:
        words=file.readlines()
    for word in words:
        stopWords.append(word)
    return stopWords

def createDataListFeatureVec(dataList,featureVocabulary):

    """
      Description:内部类生成所有文件的特征选取
      Params:
            dataList——
      Return:
            No such property: code for class: Script1
      Author:
            HY
      Modify:
            2019/4/29 21:37
    """

    def txtfeatureVec(text,featureVocabulary):
        features=[1 if word in text else 0 for word in featureVocabulary]
        return features
    dataListFeatureVec=[txtfeatureVec(ele,featureVocabulary) for ele in dataList]
    return dataListFeatureVec

def classificationResult(train_all_words_list,trainDataList,testDataList,testClassList,stopWords):
    deleteNs=range(0,500,10)
    biggestAccurate=0
    bestDeleteN=0
    resultClassList=[]
    accurates=[]
    for deleteN in deleteNs:
        featureVocabulary=createFeatureVocabulary(train_all_words_list,deleteN,stopWords)
        trainDataListFeatureVec=createDataListFeatureVec(trainDataList,featureVocabulary)
        classfier=nb.MultinomialNB().fit(trainDataListFeatureVec,trainCalssList)
        testDataListFeatureVec=createDataListFeatureVec(testDataList,featureVocabulary)
        classList=classfier.predict(testDataListFeatureVec)
        errorCount=0
        for i in range(len(classList)):
            if classList[i]!=testClassList[i]:
                errorCount+=1
        accurate=1-float(errorCount/len(testClassList))
        accurates.append(accurate)
        if biggestAccurate<accurate:
            biggestAccurate=accurate
            bestDeleteN=deleteN
            resultClassList=copy.deepcopy(classList)
    return resultClassList,biggestAccurate,bestDeleteN,deleteNs,accurates


if __name__ == '__main__':
    time1=time.time()
    train_all_words_list,trainCalssList,trainDataList=TextProcessing('./newsData/trainData')
    stopWords=getStopWordsList('./newsData/stopwords_cn.txt')
    test_all_words_list,testClassList,testDataList=TextProcessing('./newsData/testData')
    resultClassList,biggestAccurate,bestDeleteN,deleteNs,accurates=classificationResult(train_all_words_list,trainDataList,testDataList,testClassList,stopWords)
    time2=time.time()
    print('Finished in %d seconds!'%(time2-time1))
    print(resultClassList)
    print('最大的正确率是：%d%%'%(biggestAccurate*100))
    print(bestDeleteN)
    plt.figure(figsize=(8,6))
    plt.plot(deleteNs,accurates,color='blue')
    plt.title('relationship between accurate and deleteN ')
    plt.xlabel('deleteNs')
    plt.ylabel('Accurates')
    plt.show()


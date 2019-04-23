# 2个文件夹中分别含有训练集数据和测试集数据——数据已经整理好了，对测试集中的数据进行识别
# 用到KNN算法，这里不适用自己写的KNN算法分类器，使用scikit-learn库中的neighborsClassifier方法
# 思路：
# 一、读取文件，形成使用的数据结构
#     1、每一个txt文件32*32维的矩阵——换成1*1024的行向量
#     2、所有训练集中的TXT文件 就可以组合成一个大的训练矩阵
#     3、分类集标签就为整理好的文件名
#     4、类似的测试集中的数据文件也可以这样操作
# 二、数据训练，分类！
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNN
from os import listdir
import time


def switchDataToMatrix(fileName):
    """
      Description:把单个TXT文件转化为行向量
      Params:
            fileName--文件名
      Return:
            retMat——返回的矩阵（行向量）
      Author:
            HY
      Modify:
            2019/4/19 16:30
    """
    retMat=np.zeros((1,1024))
    file=open(fileName)
    for i in range(32):
        line=file.readline()#只读取一行
        for j in range(32):
            #把数据赋值给矩阵的第32*i+j列
            retMat[0,32*i+j]=int(line[j])
    return retMat

def handWrIdentiTest(trainDataDirName,testDataDirName):

    """
      Description:手写数字识别系统
      Params:
            trainDataDirName——训练集数据目录，保存的是TXT文档，这里已经把手写字迹通过软件处理成32*32维矩阵数据
            testDataDirName——测试集数据目录，保存的是TXT文档，这里已经把手写字迹通过软件处理成32*32维矩阵数据
      Return:
            classifierLabels——返回识别后的矩阵
      Author:
            HY
      Modify:
            2019/4/19 18:29
    """
    handWrLabels=[]
    classifierLabels=[]
    #获取文件夹中的所有文件，根据文件名排字典序
    train_fileList=listdir(trainDataDirName)
    m=len(train_fileList)
    trainMat=np.zeros((m,1024))
    for i in range(m):
        fileaName=train_fileList[i]
        #调用函数处理数据得到矩阵
        trainMat[i,:]=switchDataToMatrix(trainDataDirName+'/%s'%(fileaName))
        handWrLabels.append(int(fileaName.split('_')[0]))
    #sklearn中的函数，n_neighbors、algorithm一般为auto、kd_tree、ball_tree；
    #weihts函数可以选用uniform、distance
    knnNeighbor=KNN(n_neighbors=3,algorithm='ball_tree')
    #训练——猜测计算距离等
    knnNeighbor.fit(trainMat,handWrLabels)
    test_fileList=listdir(testDataDirName)
    n=len(test_fileList)
    errorCouunt=0
    for j in range(n):
        fileaName=test_fileList[j]
        testMat=switchDataToMatrix(testDataDirName+'/%s'%(fileaName))
        handWrLabel=int(fileaName.split('_')[0])
        handWrLabels.append(handWrLabel)
        #预测分类
        classifierLabel=knnNeighbor.predict(testMat)
        classifierLabels.append(classifierLabel)
        if int(classifierLabel)!=handWrLabel:
            print('真实数字是%d----预测数字是%d'%(handWrLabel,classifierLabel))
            errorCouunt+=1
    print('错误共计：%d——————总计：%d——————————————错误率是：%f %%'%(errorCouunt,len(classifierLabels),float(errorCouunt*100/len(classifierLabels))))
    return classifierLabels

if __name__ == '__main__':
    time1=time.time()
    handWrIdentiTest('trainingDigits','testDigits')
    time2=time.time()
    print('识别完成！Identification Finished in %d seconds'%(time2-time1))
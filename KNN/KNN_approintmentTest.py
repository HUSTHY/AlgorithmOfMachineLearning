import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from KNN.KNNClassify import KNNClassify




"""
函数说明：数据获取函数
parameters：
    fileName——存储数据的文件
returns：
    returnMat——前3列组成的矩阵,
    classLabelsVector——第四列数据，分类结果的向量

Modify:
    2019-04-18
"""
def fileZmartix(fileName):
    file=open(fileName)
    content=file.readlines()
    #zeros((len(content),3)) 返回一个nx3的矩阵
    returnMat=np.zeros((len(content),3))
    classLabelsVector=[]
    index=0
    for line in content:
        #去掉空白、\t 进行切片
        line=line.strip().split('\t')
        returnMat[index,:]=line[0:3]
        if line[-1]=='didntLike':
            classLabelsVector.append(1)
        if line[-1]=='smallDoses':
            classLabelsVector.append(2)
        if line[-1]=='largeDoses':
            classLabelsVector.append(3)
        index+=1
    return returnMat,classLabelsVector

"""
函数说明：数据获取函数
parameters：
    inputMatData——训练集数据矩阵
    labels——分类集矩阵
returns：


Modify:
    2019-04-18
"""
def showDataInVisualization(inputMatData,labels):
    #汉字设置；fm是matplotlib.font_manager
    myFont=fm.FontProperties(fname=r"C:\Windows\Fonts\STFANGSO.TTF")

    plt.figure(figsize=(13,7))
    labelColors=[]
    for label in labels:
        if label==1:
            labelColors.append('black')
        if label==2:
            labelColors.append('blue')
        if label==3:
            labelColors.append('red')

    #画子图，列向量选取
    axes1=plt.subplot(2,2,1)
    axes1.scatter(x=inputMatData[:,0],y=inputMatData[:,1],color=labelColors,s=10,alpha=0.5)
    axes0_title=axes1.set_title('每年飞行常客里程数与玩视频游戏时间消耗占比',FontProperties=myFont)
    axes0_xlabel=axes1.set_xlabel('每年飞行常客里程数',FontProperties=myFont)
    axes0_ylabel=axes1.set_ylabel('玩视频游戏时间消耗占比',FontProperties=myFont)
    plt.setp(axes0_title,size=9,weight='bold',color='blue')
    plt.setp(axes0_xlabel,size=7,weight='bold',color='black')
    plt.setp(axes0_ylabel,size=7,weight='bold',color='black')

    axes2=plt.subplot(2,2,2)
    axes2.scatter(x=inputMatData[:,0],y=inputMatData[:,2],color=labelColors,s=10,alpha=0.5)
    axes1_title=axes2.set_title('每年飞行常客里程数与每周消耗冰淇淋公升数',FontProperties=myFont)
    axes1_xlabel=axes2.set_xlabel('每年飞行常客里程数',FontProperties=myFont)
    axes1_ylabel=axes2.set_ylabel('周消耗冰淇淋公升数',FontProperties=myFont)
    plt.setp(axes1_title,size=9,color='blue')
    plt.setp(axes1_xlabel,size=7,color='black')
    plt.setp(axes1_ylabel,size=7,color='black')

    axes3=plt.subplot(2,2,3)
    axes3.scatter(x=inputMatData[:,1],y=inputMatData[:,2],color=labelColors,s=10,alpha=0.5)
    axes2_title=axes3.set_title('玩视频游戏时间消耗占比与每周消耗冰淇淋公升数',FontProperties=myFont)
    axes2_xlabel=axes3.set_xlabel('每年飞行常客里程数',FontProperties=myFont)
    axes2_ylabel=axes3.set_ylabel('周消耗冰淇淋公升数',FontProperties=myFont)
    plt.setp(axes2_title,size=9,weight='bold',color='blue')
    plt.setp(axes2_xlabel,size=7,weight='bold',color='black')
    plt.setp(axes2_ylabel,size=7,weight='bold',color='black')


    #图例设置
    didntlike=plt.scatter([],[],color='black',label='didntlike')
    smallDoses=plt.scatter([],[],color='blue',label='smallDoses')
    largeDoses=plt.scatter([],[],color='red',label='largeDoses')

    # 图例添加
    axes1.legend(handles=[didntlike,smallDoses,largeDoses])
    axes2.legend(handles=[didntlike,smallDoses,largeDoses])
    axes3.legend(handles=[didntlike,smallDoses,largeDoses])
    plt.show()

"""
函数说明：数据获取函数
parameters：
    dataSet——训练集初始数据矩阵
returns：
    normalDatasSet——归一化的训练集数据矩阵
    minVal——训练集中的每列最小值组成的向量矩阵
    ranges——范围区间

Modify:
    2019-04-18
"""
def dataNormal(dataSet):
    # min(0)返回该矩阵中每一列的最小值
    # min(1)返回该矩阵中每一行的最小值
    # max(0)返回该矩阵中每一列的最大值
    # max(1)返回该矩阵中每一行的最大值
    #这里采取的归一化方法  newValue = (oldValue - min) / (max - min)
    maxVal=dataSet.max(0)
    minVal=dataSet.min(0)
    ranges=maxVal-minVal
    normalDatasSet=dataSet-np.tile(minVal,(np.shape(dataSet)[0],1))
    normalDatasSet=normalDatasSet/np.tile(maxVal,(np.shape(dataSet)[0],1))
    return normalDatasSet,minVal,ranges

"""
函数说明：测试KNN算法
parameters：
    
returns：


Modify:
    2019-04-18
"""
def testKNNAlgorithm():
    hoTatio=0.10
    datingDataMat,datingLabes=fileZmartix('datingTestSet.txt')
    normalDatasSet,minVal,ranges=dataNormal(datingDataMat)
    m=np.shape(normalDatasSet)[0]
    numTestVec=int(m*hoTatio)
    errorCount=0
    #取numTestVec个行向量
    for i in range(numTestVec):
        #normalDatasSet[i,:].reshape(1,np.shape(normalDatasSet)[1])取第i行形成一个行向量——为了统一KNN算法中的数据结构
        #classResultLabel是一个list 其中只有一个元素，因为KNN算法中的测试数据只有一个行向量
        classResultLabel=KNNClassify(normalDatasSet[i,:].reshape(1,np.shape(normalDatasSet)[1]),normalDatasSet[numTestVec:m,:],datingLabes[numTestVec:m],4,0)
        if classResultLabel[0]!=datingLabes[i]:
            print("分类结果：%d\t真实类别是：%d"%(classResultLabel[0],datingLabes[i]))
            errorCount+=1
    print("错误率为：%f%%"%(errorCount*100/(float(numTestVec))))


"""
函数说明：使用KNN算法对具有3维特征的人进行分类
parameters： 
returns：
Modify:
    2019-04-18
"""
def classifyAnyPerson():
    resultClassList=['不喜欢','有一点好感','好感度爆满']
    Flytime=float(input("每年飞行时间（单位小时）："))
    gameTimeRate=float(input("游戏时间占比："))
    iceCream=float(input("每周吃的冰淇淋（单位升）："))
    #假如不强制转换的话 数据类型是str，转了才是float型，才能和下面的矩阵做运算
    personData=np.array([Flytime,gameTimeRate,iceCream])
    fileName='datingTestSet.txt'
    #提取训练集数据
    dataMat,labelsVector=fileZmartix(fileName)
    #训练集归一化，并返回结果
    normalDataSet,minVal,Ranges=dataNormal(dataMat)
    #测试集归一化
    # personDataNormal=np.divide((personData-minVal),Ranges)
    personDataNormal=(personData-minVal)/Ranges
    personClassifyResult=KNNClassify(personDataNormal,normalDataSet,labelsVector,4,0)
    print("海伦对这个样的的印象是：%s"%resultClassList[personClassifyResult[0]-1])



if __name__ == '__main__':
    returnMat,classLabelsVector=fileZmartix('datingTestSet.txt')
    showDataInVisualization(returnMat,classLabelsVector)
    normalDatasSet,minVal,ranges=dataNormal(returnMat)
    testKNNAlgorithm()
    classifyAnyPerson()
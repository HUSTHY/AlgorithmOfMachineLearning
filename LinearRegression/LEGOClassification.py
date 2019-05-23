#!/usr/bin/env python

# -*- encoding: utf-8 -*-

'''
@Author  :    HY
@Software:   PyCharm
@File    :   LEGOClassification.py
@Time    :   2019/5/23 15:48
@Desc    :

'''

from bs4 import BeautifulSoup
from sklearn.linear_model import Ridge
import numpy as np
import matplotlib.pyplot as plt
def scrapePage(retX, retY, inFile, yr, numPce, origPrc):
    """
        Description:
        Params:
                
        Return:
                
        Author:
                HY
        Modify:
                2019/5/23 21:35
    """
    # 打开并读取HTML文件
    with open(inFile, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # 根据HTML页面结构进行解析
    currentRow = soup.find_all('table', r="%d" % i)
    while (len(currentRow) != 0):
        currentRow = soup.find_all('table', r="%d" % i)
        title = currentRow[0].find_all('a')[1].text
        titles=currentRow[0].find_all('a')
        lwrTitle = title.lower()
        # 查找是否有全新标签
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        # 查找是否已经标志出售，我们只收集已出售的数据
        soldUnicde = currentRow[0].find_all('td')[3].find_all('span')
        if len(soldUnicde) == 0:
            print("商品 #%d 没有出售" % i)
        else:
            # 解析页面获取当前价格
            soldPrice = currentRow[0].find_all('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$', '')
            priceStr = priceStr.replace(',', '')
            if len(soldPrice) > 1:
                priceStr = priceStr.replace('Free shipping', '')
            sellingPrice = float(priceStr)
            # 去掉不完整的套装价格
            if sellingPrice > origPrc * 0.5:
                # print("%d\t%d\t%d\t%f\t%f" % (yr, numPce, newFlag, origPrc, sellingPrice))
                retX.append([yr, numPce, newFlag, origPrc])
                retY.append(sellingPrice)
        i += 1
        currentRow = soup.find_all('table', r="%d" % i)


def setDataCollect(retX, retY):
    """
    函数说明:依次读取六种乐高套装的数据，并生成数据矩阵
    Parameters:
        无
    Returns:
    
    Modify:
    """
    scrapePage(retX, retY, './lego/lego8288.html', 2006, 800, 49.99)  # 2006年的乐高8288,部件数目800,原价49.99
    scrapePage(retX, retY, './lego/lego10030.html', 2002, 3096, 269.99)  # 2002年的乐高10030,部件数目3096,原价269.99
    scrapePage(retX, retY, './lego/lego10179.html', 2007, 5195, 499.99)  # 2007年的乐高10179,部件数目5195,原价499.99
    scrapePage(retX, retY, './lego/lego10181.html', 2007, 3428, 199.99)  # 2007年的乐高10181,部件数目3428,原价199.99
    scrapePage(retX, retY, './lego/lego10189.html', 2008, 5922, 299.99)  # 2008年的乐高10189,部件数目5922,原价299.99
    scrapePage(retX, retY, './lego/lego10196.html', 2009, 3263, 249.99)  # 2009年的乐高10196,部件数目3263,原价249.99

def sklearnRidgeRegression(x,y,al):
    """
        Description:sklearn岭回归
        Params:

        Return:

        Author:
                HY
        Modify:
                2019/5/23 19:39
    """
    ridge=Ridge(alpha=al)
    ridge.fit(x,y)
    ws = ridge.coef_#得到权重不包含w0,w0=0的，然后那个截距ridge.intercep
    return ws

def showWs(WsMat):
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['FangSong']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10,7))
    #画一个矩阵m*n 就会得到那条曲线；每一条曲线表示的是矩阵每一列的值随着序号增大的变化
    plt.plot(WsMat)
    plt.title('乐高玩具价格预测——岭回归算法，回归系数和alpha的关系')
    plt.xlabel('alpha——np.exp(i-15)')
    plt.ylabel('回归系数')
    plt.show()

if __name__ == '__main__':
    lgX = []
    lgY = []
    setDataCollect(lgX, lgY)
    n=np.shape(lgX)[1]
    WMat=np.zeros((30,n))
    for i in range(30):
        al=np.exp(i-15)
        WMat[i,:]=sklearnRidgeRegression(lgX,lgY,al)
        print(WMat[i,:])
    showWs(WMat)


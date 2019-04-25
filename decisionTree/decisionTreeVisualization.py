from sklearn.preprocessing import LabelEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pandas as pd
import pydotplus

"""
  Description:决策树的可视化 用GraphViz+pydotPl库，树的生成还是用的sklearn的tree.DecisionTreeClassifier
  Author: 
        HY
        
  Modify: 
        2019/4/24 12:38
"""


if __name__ == '__main__':
    with open('lenses.txt') as file:
        lenses=[f.strip().split('\t') for f in file.readlines()]
    targetList=[]
    for l in lenses:
        targetList.append(l[-1])
    labels=['age','prescript症状','astigmatio散光','tearRate眼泪数量']
    lenseDic={}
    for label in labels:
        list=[]
        for elem in lenses:
            list.append(elem[labels.index(label)])
        lenseDic[label]=list
    lenses_pd=pd.DataFrame(lenseDic)
    labelEnc=LabelEncoder()
    for col in lenses_pd.iteritems():
        lenses_pd[col[0]]=labelEnc.fit_transform(col[1])
    decisionTree=tree.DecisionTreeClassifier(criterion='entropy')
    decisionTree.fit(lenses_pd.values.tolist(),targetList)
    #上面是数据预处理（只有把String型的换为float或者int型的，sklearn才能识别）和树的生成
    # 下面就是树的可视化部分，用到了sklearn的tree.export_graphviz()和pydotplus的graph_from_dot_data()
    dot_data=StringIO()
    tree.export_graphviz(decisionTree,out_file=dot_data,max_depth=100,
                        feature_names=lenses_pd.keys(),class_names=decisionTree.classes_,
                         filled=True,rounded=True,special_characters=True)
    graphTree=pydotplus.graph_from_dot_data(dot_data.getvalue().replace('helvetica','"Microsoft YaHei"'))#replace后面是为了把helvetica修改为微软的字体
    graphTree.write_pdf('decisionTree.pdf')#树保存为PDF格式文件


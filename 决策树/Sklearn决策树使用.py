# -*- coding: UTF-8 -*-
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus


def getDataSet(filename):
    with open(filename, 'r') as f:
        data = [i.strip().split('\t') for i in f.readlines()]
    return data


if __name__ == '__main__':
    data = getDataSet('隐形眼镜数据集.txt')
    data_target = []
    # 得到已有的数据标签集
    for each in data:
        data_target.append(each[-1])
    dataLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    data_list = []
    data_dict = {}
    for each_label in dataLabels:
        for each in data:
            data_list.append(each[dataLabels.index(each_label)])
        data_dict[each_label] = data_list
        data_list = []
    lenses_pd = pd.DataFrame(data_dict)
    # 创建LabelEncoder()对象，用于序列化
    le = LabelEncoder()
    # 为每一列序列化
    for col in lenses_pd.columns:
        lenses_pd[col] = le.fit_transform(lenses_pd[col])
    # 创建DecisionTreeClassifier()类
    clf = tree.DecisionTreeClassifier(max_depth=4)
    # 构建决策树
    clf = clf.fit(lenses_pd.values.tolist(), data_target)
    dot_data = StringIO()
    tree.export_graphviz(clf, out_file=dot_data, feature_names=lenses_pd.keys(),
                         class_names=clf.classes_, filled=True, rounded=True,special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("tree.pdf")
    # 尝试预测
    print(clf.predict([[1, 1, 1, 0]]))

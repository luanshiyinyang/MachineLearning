# -*-coding:utf-8 -*-
import numpy as np
import scipy as sp
from sklearn import tree
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def getData():
    '''
    获取数据集
    :return:
    '''
    data = []
    labels = []
    with open("data/1.txt") as ifile:
            for line in ifile:
                tokens = line.strip().split(' ')
                data.append([float(tk) for tk in tokens[:-1]])
                labels.append(tokens[-1])
    x = np.array(data)
    labels = np.array(labels)
    y = np.zeros(labels.shape)
    y[labels == 'fat'] = 1
    return x, y


if __name__ == '__main__':
    x, y = getData()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    # 使用信息熵作为划分标准，对决策树进行训练
    clf = tree.DecisionTreeClassifier(criterion='entropy')
    print(clf)
    clf.fit(x_train, y_train)
    with open("tree.dot", 'w') as f:
        f = tree.export_graphviz(clf, out_file=f)
    # 系数反映每个特征的影响力。越大表示该特征在分类中起到的作用越大 '''
    print('两个特征所占的权重是：', clf.feature_importances_)
    # 测试结果显示
    answer = clf.predict(x_test)
    print('测试数据是：', x_test)
    print('测试数据使用模型预测对应的类是：', answer)
    print('测试数据对应的类是：', y_test)
    print(np.mean(answer == y_test))
    # 准确率与召回率'''
    precision, recall, thresholds = precision_recall_curve(y_train, clf.predict(x_train))
    answer = clf.predict_proba(x)[:, 1]
    print(classification_report(y, answer, target_names=['thin', 'fat']))
    # 落地模型
    import pydotplus
    dot_data = tree.export_graphviz(clf, out_file=None)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("tree.pdf")

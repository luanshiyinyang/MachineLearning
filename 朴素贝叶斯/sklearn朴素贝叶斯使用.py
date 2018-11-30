# -*- coding:UTF-8 -*-
"""
这里演示朴素贝叶斯的使用
"""
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def getDataSet(fileName):
    # 读取数据
    dataSet = []
    label = []
    with open(fileName) as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        line = line.split('\t')
        dataSet.append(line[:3])
        label.append(line[-1])
    return dataSet, label


def scale(dataSet):
    '''
    归一化
    :param dataSet: 数据集
    :return:
    '''
    scaler = MinMaxScaler()
    dataSet_new = scaler.fit_transform(dataSet)
    return dataSet_new


def divide(dataSet, labels):
    '''
    分类数据，按比例拆开
    :param dataSet:
    :param labels:
    :return:
    '''
    train_data, test_data, train_label, test_label = train_test_split(dataSet, labels, test_size=0.2)
    return train_data, test_data, train_label, test_label


if __name__ == '__main__':
    file = './data/datingSet.txt'
    data, labels = getDataSet(file)
    data = scale(data)
    train_data, test_data, train_label, test_label = divide(data, labels)
    # 创建高斯分布模型
    model = GaussianNB()
    model.fit(train_data, train_label)
    data_predicted = model.predict(test_data)
    mess = metrics.classification_report(test_label, data_predicted)
    print(mess)

    label_nos = list(set(labels))
    # 在之前的代码混淆矩阵可视化我是自己写的
    mess2 = metrics.confusion_matrix(test_label, data_predicted, labels=label_nos)
    print(mess2)



# -*- coding:UTF-8 -*-
"""
sklearn中Logistic回归的使用
"""
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler



def solve():
    '''
    处理原始数据
    原始数据集有缺失，经过处理
    特征缺失换0，不影响，因为Logistic回归中立
    标签缺失删除
    :return:
    '''
    rawFile = open('./data/rawdata.txt')
    dataSet = []
    label = []
    print(rawFile)
    for line in rawFile:
        line = line.strip()
        line = line.replace("\n", '')
        line2list = line.split(' ')

        if line2list[-1] == "?":
            pass
        else:
            del line2list[2]
            del line2list[23]
            del line2list[23]
            for i in range(len(line2list)):
                if line2list[i] == "?":
                    line2list[i] = 0
            dataSet.append(line2list[:len(line2list)-2])
            label.append(line2list[-1])
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
    比例切分两个数据集
    :param dataSet:
    :param labels:
    :return:
    '''
    train_data, test_data, train_label, test_label = train_test_split(dataSet, labels, test_size=0.2)
    return train_data, test_data, train_label, test_label


if __name__ == '__main__':
    data, labels = solve()
    data = scale(data)

    # 得到数据集
    trainDataSet, testDataSet, trainDataLabel, testDataLabel = divide(data, labels)
    # 建立模型并训练
    # 其中，solver表示优化算法选择参数，有五个可选参数
    # newton-cg,lbfgs,liblinear,sag,saga
    # 默认为liblinear
    # solver参数决定了我们对逻辑回归损失函数的优化方法
    # 具体如何选择可以查看官方文档，一般小数据集使用下面使用的即可（多分类不是）
    # 其中，max_iter指明最大迭代次数，默认为10，一般无效
    classifier = LogisticRegression(solver='liblinear', max_iter=10).fit(trainDataSet, trainDataLabel)
    # 给出分类准确率，指明数据集和实际标签即可
    rst = classifier.score(testDataSet, testDataLabel) * 100.0
    print('正确率:{:.2f}'.format(rst))


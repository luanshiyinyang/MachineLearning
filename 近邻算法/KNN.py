# -*- coding: UTF-8 -*-
import numpy as np
import operator

def getDataSet():
    """

    :return: 样本集数据(包括属性和标签）
    """
    # 六个训练集数据
    data = np.array([[1, 2, 3], [4, 5, 6], [3, 6, 7], [2, 3, 1], [5, 8, 10], [-1, 2, 9]])

    label = np.array([["A"], ["A"], ["B"], ["C"], ["B"], ["A"]])
    return data, label


def classification(testSet, sampleSet, samplelabel, k):
    """
    :param testSet: 测试集，待分类的数据集
    :param sampleSet: 样本集，用于比对的训练集
    :param samplelabel: 样本集对应标签，也可放在样本集合内作为一个属性
    :param k: k近邻的参数，最接近的k个数据
    :return: 分类好的结果集
    """
    # 返回矩阵行数,也就是此时训练集数据个数
    sampleDataNumber = sampleSet.shape[0]
    rstList = []
    for item in testSet:
        Mat = np.tile(item, (sampleDataNumber, 1))
        # 利用矩阵运算，直接求得与每个样本数据的距离
        rstMat = Mat - sampleSet
        rstMat = rstMat**2
        # 参数为空（所有元素求和），参数为0（列求和），参数为1（行求和）
        sumMat = rstMat.sum(axis=1)
        distance = sumMat**0.5
        print(distance)
        # 返回升序索引列表
        sortedIndex = distance.argsort()
        print(sortedIndex)
        # 由此得知前k个元素的下标
        rst_dict = dict()
        for i in range(k):
            currentLabel = samplelabel[sortedIndex[i]][0]
            print(currentLabel)
            rst_dict[currentLabel] = rst_dict.get(currentLabel, 0) + 1
        # 降序排列
        sortedResult = sorted(rst_dict.items(), key=operator.itemgetter(1), reverse=True)
        rst = sortedResult[0][0]
        temp = list()
        temp.append(rst)
        rstList.append(temp)
    return np.array(rstList)


if __name__ == '__main__':
    # 创建数据集
    data, label = getDataSet()
    print("样本集")
    print(data)
    print(label)
    # 测试集
    test = np.array([[1, 2, 3], [10, 5, 7]])
    print("测试集")
    print(test.shape)
    print(test)
    rst = classification(testSet=test, sampleSet=data, samplelabel=label, k=3)
    print(rst)
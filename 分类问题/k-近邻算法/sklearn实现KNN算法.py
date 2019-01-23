# -*-coding:UTF-8 -*-
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from numpy import *


def file2matrix(filename):
    '''
    数据读取为矩阵
    :param filename: 文件名
    :return: 特征矩阵， 标签矩阵
    '''
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector


def autoNorm(dataSet):
    '''
    归一化数据
    :param dataSet: 数据集
    :return: 归一化后的数据集
    '''
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    min_max_scaler = MinMaxScaler()
    normDataSet = min_max_scaler.fit_transform(dataSet)
    return normDataSet, ranges, minVals


def classifyPerson(model):
    '''
    网站实测
    :return: 可能标签
    '''
    resultList = ['不喜欢', '一般喜欢', '特别喜欢']
    percentTats = float(input("玩游戏占的百分比"))
    ffMiles = float(input("每年坐飞机多少公里"))
    iceCream = float(input("每年吃多少公升的冰淇淋"))
    datingDataMat, datingLabels = file2matrix('data/datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([[ffMiles, percentTats, iceCream]])
    model.fit(normMat, datingLabels)
    print((inArr - minVals) / ranges)
    rst = model.predict((inArr - minVals) / ranges)
    print("你将有可能对这个人是:", resultList[int(rst) - 1])


if __name__ == '__main__':
    model = KNeighborsClassifier(n_neighbors=3, weights='uniform', algorithm='auto', leaf_size=30,
                                 p=2, metric='minkowski', metric_params=None, n_jobs=1)
    classifyPerson(model)


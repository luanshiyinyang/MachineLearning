# -*-coding:utf-8-*-
from numpy import *
import operator


def classify0(inX, dataSet, labels, k):
    '''
    利用距离公式计算出最接近的k个样本中出现最多的标签
    :param inX: 待分类的样本
    :param dataSet: 特征矩阵
    :param labels: 类别矩阵
    :param k: k值
    :return: 最接近的标签
    '''
    print(inX)
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDisttances = sqDiffMat.sum(axis=1)
    distances = sqDisttances ** 0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 依照标签出现的频率排序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    # 选取频率最大的标签
    return sortedClassCount[0][0]


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
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    print(normDataSet)
    return normDataSet, ranges, minVals


def classifyPerson():
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
    inArr = array([ffMiles, percentTats, iceCream])
    print((inArr - minVals) / ranges)
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("你将有可能对这个人是:", resultList[int(classifierResult) - 1])


if __name__ == '__main__':
    classifyPerson()

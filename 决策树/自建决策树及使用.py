# -*- coding: UTF-8 -*-
from math import log
import numpy as np
import operator
import pickle
from matplotlib.font_manager import FontProperties
import matplotlib.pyplot as plt
featLabels = []


def caculateShannonEnt(dataSet):
    '''

    :param dataSet: 传入的数据集
    :return: 返回数据集的香农熵
    '''
    # 获取数据的记录数，对矩阵而言就是行数
    numOfData = len(dataSet)
    # 标签出现次数的字典
    labelCountDict = {}
    for item in dataSet:
        currentLabel = item[-1]
        labelCountDict[currentLabel] = labelCountDict.get(currentLabel, 0) + 1
    ShannonEntropy = 0.0
    # 遍历求和计算香农熵
    for key in labelCountDict.keys():
        prob = float(labelCountDict[key]) / numOfData
        ShannonEntropy -= prob * log(prob, 2)
    return ShannonEntropy


def splitDataSet(dataSet, axis, value):
    '''
    返回第axis个特征为value的数据集
    :param dataSet: 数据集
    :param axis: 子数据集的属性下标
    :param value: 子数据集该下标属性的取值
    :return: 满足条件的字数据集
    '''
    retDataSet = []
    # 创建返回的数据集列表
    for featVec in dataSet.tolist():
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return np.array(retDataSet)


def maxInfoGain(dataSet):
    '''

    :param dataSet: 数据集
    :return: 最大信息增益属性的下标
    '''
    # ndarray的shape保存行列信息，1下标对应列数目，也就是每行属性个数
    numOfFeatures = dataSet.shape[1] - 1
    # 计算香农熵
    ShannonEntropy = caculateShannonEnt(dataSet)
    # 最大信息增益初始化
    maxInfoG = 0.0
    maxInfoGIndex = -1
    for i in range(numOfFeatures):
        # 所有数据的第i个特征
        subArray = [rst[i] for rst in dataSet]
        # 比如15个数据，A属性取值只有1,2,3，那么uniqueArray结果只有1,2,3
        uniqueArray = set(subArray)
        # 经验香农熵
        expeEntropy = 0.0
        for value in uniqueArray:
            # 取出第i个特征为value的数据集合的子集
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / float(len(dataSet))  # 计算子集的概率
            expeEntropy += prob * caculateShannonEnt(subDataSet)  # 根据公式计算经验条件熵
        # 信息增益
        InfoG = ShannonEntropy - expeEntropy
        # print("第%d个特征的增益为%.3f" % (i, InfoG))
        if InfoG > maxInfoG:
            maxInfoG = InfoG
            maxInfoGIndex = i
    return maxInfoGIndex


def maxCount(labelList):
    '''

    :param labelList:所有数据的标签列表
    :return:出现次数最多的标签
    '''
    classCountDict = {}
    for vote in labelList:
        if vote not in classCountDict.keys():
            classCountDict[vote] = 0
        classCountDict[vote] += 1
    sortedClassCount = sorted(classCountDict.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def createTree(dataSet, labels, featLabels):
    '''
    递归创建决策树字典
    :param dataSet: 数据集
    :param labels:
    :param featLabels:
    :return:
    '''
    # 取出所有标签
    labelList = [rst[-1] for rst in dataSet]
    # 如果类别相同则停止划分
    if labelList.count(labelList[0]) == len(labelList):
        return labelList[0]
    # 遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1:
        return maxCount(labelList)
    # 选择最优特征
    bestFeat = maxInfoGain(dataSet)
    bestFeatLabel = labels[bestFeat]
    featLabels.append(bestFeatLabel)
    # 根据最优特征的标签生成树
    myTree = {bestFeatLabel: {}}
    # 删除已经使用特征标签
    del(labels[bestFeat])
    # 得到训练集中所有最优特征的属性值
    featValues = [rst[bestFeat] for rst in dataSet]
    # 去掉重复的属性值
    uniqueVals = set(featValues)
    # 遍历特征，创建决策树。
    for value in uniqueVals:
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value), labels, featLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    '''

    :param inputTree: 生成的决策树
    :param featLabels: 存储选择的最优特征标签
    :param testVec:测试数据列表
    :return:
    '''
    classLabel = None
    firstStr = next(iter(inputTree))
    # 获取决策树结点
    secondDict = inputTree[firstStr]
    # 下一个字典
    firstStr = next(iter(inputTree))
    featIndex = featLabels.index(firstStr)
    for key in list(secondDict.keys()):
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


def storeTree(inputTree, filename):
    '''
    本地落地决策树
    :param inputTree:
    :param filename:
    :return:
    '''

    with open(filename, 'wb') as fw:
        pickle.dump(inputTree, fw)


def grabTree(filename):
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == '__main__':
    # 测试数据
    dataSet = [[0, 0, 0, 0, 'no'], [0, 0, 0, 1, 'no'], [0, 1, 0, 1, 'yes'],
               [0, 1, 1, 0, 'yes'], [0, 0, 0, 0, 'no'], [1, 0, 0, 0, 'no'],
               [1, 0, 0, 1, 'no'], [1, 1, 1, 1, 'yes'], [1, 0, 1, 2, 'yes'],
               [1, 0, 1, 2, 'yes'], [2, 0, 1, 2, 'yes'], [2, 0, 1, 1, 'yes'],
               [2, 1, 0, 1, 'yes'], [2, 1, 0, 2, 'yes'], [2, 0, 0, 0, 'no']]
    labels = ['属性A', '属性B', '属性C', '属性D', ]
    dataSet = np.array(dataSet)
    tree = createTree(dataSet, labels, featLabels)
    print(tree)
    storeTree(tree, 'ZCTree')
    testData = ['0', '1']
    rst = classify(tree, featLabels, testData)
    print(rst)

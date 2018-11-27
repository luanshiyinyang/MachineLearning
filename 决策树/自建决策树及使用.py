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


# 返回第axis个特征为value的数据集
def splitDataSet(dataSet, axis, value):
    '''

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
    递归创建决策树
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


def getNumLeafs(myTree):
    '''
    得到决策树所有的叶子节点
    :param myTree: 字典型决策树
    :return: 叶子节点数
    '''
    numLeafs = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    '''

    :param myTree: 字典型决策树
    :return: 决策树层数
    '''
    maxDepth = 0
    firstStr = next(iter(myTree))
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth: maxDepth = thisDepth
    return maxDepth


def createPlot(inTree):
    '''

    :param inTree:
    :return:
    '''
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    # 去掉x、y轴
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    # x偏移
    plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    '''
    绘制节点
    :param nodeTxt:节点名
    :param centerPt:文本位置
    :param parentPt:剪头位置
    :param nodeType:节点格式
    :return:
    '''
    # 定义箭头格式
    arrow_args = dict(arrowstyle="<-")
    font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
    createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction', xytext=centerPt,
                            textcoords='axes fraction', va="center", ha="center",
                            bbox=nodeType, arrowprops=arrow_args, FontProperties=font)


def plotMidText(cntrPt, parentPt, txtString):
    '''
    标注有向边属性值
    :param cntrPt: 计算标注位置
    :param parentPt:
    :param txtString:标注的内容
    :return:
    '''
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    createPlot.ax1.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)


def plotTree(myTree, parentPt, nodeTxt):
    '''
    绘制决策树
    :param myTree: 决策树
    :param parentPt: 标注内容
    :param nodeTxt:节点名
    :return:
    '''
    # 设置结点格式
    decisionNode = dict(boxstyle="sawtooth", fc="0.8")
    # 设置叶结点格式
    leafNode = dict(boxstyle="round4", fc="0.8")
    # 获取决策树叶结点数目，决定了树的宽度
    numLeafs = getNumLeafs(myTree)
    # 获取决策树层数
    depth = getTreeDepth(myTree)
    # 下个字典
    firstStr = next(iter(myTree))
    # 中心位置
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW, plotTree.yOff)
    # 标注有向边属性值
    plotMidText(cntrPt, parentPt, nodeTxt)
    # 绘制结点
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    # 下一个字典，也就是继续绘制子结点
    secondDict = myTree[firstStr]
    # y偏移
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            # 不是叶结点，递归调用继续绘制
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # 如果是叶结点，绘制叶结点，并标注有向边属性值
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff), cntrPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    '''
    创建绘制面板
    :param inTree:
    :return:
    '''
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5,1.0), '')
    plt.show()


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
    storeTree(tree, 'ZCTree')
    createPlot(tree)
    testData = ['0', '1']
    rst = classify(tree, featLabels, testData)
    print(rst)

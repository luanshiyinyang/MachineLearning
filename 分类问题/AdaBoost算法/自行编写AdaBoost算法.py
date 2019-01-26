# -*- coding:utf-8 -*-
from numpy import *
import matplotlib.pyplot as plt


def loadSimpData():
    '''
    创建一个简单数据集
    :return:
    '''
    datMat = matrix([[1., 2.1], [2.,  1.1], [1.3,  1.], [1.,  1.], [2.,  1.]])
    classLabels = [1.0, 1.0, -1.0, -1.0, 1.0]
    return datMat, classLabels


def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    '''
    对数据进行分类
    通过阈值比对将数据分类，所有在阈值一边的会分到类别1中，而在另一边的会分到类别-1中
    :param dataMatrix:
    :param dimen:
    :param threshVal:
    :param threshIneq:
    :return:
    '''
    retArray = ones((shape(dataMatrix)[0], 1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:, dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:, dimen] > threshVal] = -1.0
    return retArray


def buildStump(dataArr,classLabels,D):
    '''
    找到最佳决策树
    遍历stumpClassify()函数所有可能的输入值，并找到基于数据权重向量D的单层决策树。
    :param dataArr:
    :param classLabels:
    :param D:
    :return:
    '''
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m, n = shape(dataMatrix)
    # 用于在特征空间的所有可能值上进行遍历
    numSteps = 10.0
    # 存储给定权重向量D时所得到的单层决策树信息
    bestStump = {}
    bestClasEst = mat(zeros((m, 1)))
    # 记录可能的最小错误率
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:, i].min()
        rangeMax = dataMatrix[:, i].max()
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1, int(numSteps) + 1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix, i, threshVal, inequal)
                errArr = mat(ones((m, 1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T * errArr
                print('split：dim %d,thresh %.2f,thresh ineqal:%s,the weighted error is %.3f' % (i, threshVal, inequal, weightedError))
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump, minError, bestClasEst


def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    '''
    完整的训练算法
    :param dataArr:
    :param classLabels:
    :param numIt:
    :return:
    '''
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m, 1))/m)
    aggClassEst = mat(zeros((m, 1)))
    for i in range(numIt):
        bestStump, error, classEst = buildStump(dataArr, classLabels, D)
        print("D:", D.T)
        alpha = float(0.5*log((1.0-error)/max(error, 1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print("classEst: ", classEst.T)
        # 计算下次迭代的新权重D
        expon = multiply(-1*alpha*mat(classLabels).T, classEst)
        D = multiply(D, exp(expon))
        D = D/D.sum()
        # 计算累加错误率
        aggClassEst += alpha*classEst
        print("aggClassEst: ", aggClassEst.T)
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))
        errorRate = aggErrors.sum()/m
        print("total error: ", errorRate)
        if errorRate == 0.0:
            break
    return weakClassArr, aggClassEst


def adaClassify(datToClass,classifierArr):
    '''
    实现分类函数，测试算法
    :param datToClass:
    :param classifierArr:
    :return:
    '''
    dataMatrix = mat(datToClass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m, 1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[0][i]['dim'], classifierArr[0][i]['thresh'], classifierArr[0][i]['ineq'])
        aggClassEst += classifierArr[0][i]['alpha']*classEst
        print(aggClassEst)
    return sign(aggClassEst)


def loadDataSet(fileName):
    '''
    在难数据集上应用
    自适应数据加载函数
    :param fileName:
    :return:
    '''
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat, labelMat


def plotROC(predStrengths, classLabels):
    '''
    绘制ROC曲线
    :param predStrengths:
    :param classLabels:
    :return:
    '''
    cur = (1.0, 1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels)-numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0], cur[0]-delX], [cur[1], cur[1]-delY], c='b')
        cur = (cur[0]-delX, cur[1]-delY)
    ax.plot([0, 1], [0, 1], 'b--')
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve for AdaBoost horse colic detection system')
    ax.axis([0, 1, 0, 1])
    plt.show()
    print("the Area Under the Curve is: ", ySum*xStep)


if __name__ == '__main__':
    dataMat, classLabels = loadSimpData()
    classifierArr = adaBoostTrainDS(dataMat, classLabels, 30)
    adaClassify([[5, 5], [0, 0]], classifierArr)

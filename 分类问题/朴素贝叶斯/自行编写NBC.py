# -*- coding:UTF-8 -*-
from numpy import *


# 包含所有文档 不含重复词的list
def createVocabList(dataSet):
    '''
    包含所有文档，不含重复词的一个list
    :param dataSet:
    :return:
    '''
    vocabSet = set([])
    for document in dataSet:
        # 求并集
        vocabSet = vocabSet | set(document)
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    '''
    判断某个词条是否在文档中出现
    :param vocabList: 词汇表
    :param inputSet: 文档
    :return:
    '''
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print("单词: %s 不在我的词汇里面!" % word)
    return returnVec


def trainNB0(trainMatrix, trainCategory):
    '''
    朴素贝叶斯训练函数
    :param trainMatrix: 训练集
    :param trainCategory:
    :return:
    '''
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    p0Num = ones(numWords)
    p1Num = ones(numWords)
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive


def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    '''
    给定词向量，判断类别
    :param vec2Classify:
    :param p0Vec:
    :param p1Vec:
    :param pClass1:
    :return:
    '''
    p1 = sum(vec2Classify*p1Vec)+log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+log(1.0-pClass1)
    if p1 > p0:
        return 1
    else:
        return 0


def textParse(bigString):
    '''
    字符串解析
    :param bigString: 传入的大字符串，一般是整个邮件内容的字符串
    :return: 字符串列表，并删除小于两个字符的干扰字符串，大写转为小写
    '''
    import re
    # 接收一个大字符串并将其解析为字符串列表
    listOfTokens = re.split('x*', bigString)
    # 去掉少于两个的字符串并全部转化为小写
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]


def spamTest():
    '''
    过滤功能测试
    :return:
    '''
    docList = []
    classList = []
    fullText = []
    for i in range(1, 26):
        wordList = textParse(open('data/spam/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(open('data/ham/%d.txt' % i, "rb").read().decode('GBK', 'ignore'))
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(array(trainMat), array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("分类错误的是： %s" % vocabList[docIndex])
    print('错误率是:', float(errorCount)/len(testSet))


if __name__ == '__main__':
    spamTest()

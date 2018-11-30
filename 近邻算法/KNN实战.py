# -*- coding: UTF-8 -*-
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from KNN import classification


def parseFile(filename):
    '''
    解析文件
    :param filename: 解析的文件名称（含路径）
    :return: 特征矩阵及标签集
    '''
    # 读取无表头文件
    data = pd.read_excel(filename, header=None)
    sampleSet = data.iloc[:, :3]
    sampleLabel = data.iloc[:, 3]
    # 构造样本集合标签集
    sampleMat = sampleSet.values
    sampleLabelMat = np.array([sampleLabel.values]).T
    for i in range(len(sampleLabelMat)):
        if sampleLabelMat[i][0] == "讨厌":
            sampleLabelMat[i][0] = 0
        elif sampleLabelMat[i][0] == "喜欢":
            sampleLabelMat[i][0] = 1
        else:
            sampleLabelMat[i][0] = 2
    return sampleMat, sampleLabelMat


def showDatas(data, label):
    '''
    :param data: 样本集
    :param label: 标签集
    :return: None
    '''
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    # 画布被划分成为2*2区域，此时axs[0][0]代表第一个区域
    fig, axs = plt.subplots(nrows=2, ncols=2, sharex=False, sharey=False, figsize=(13, 8))
    labelColor = []
    for i in label:
        if i[0] == 0:
            labelColor.append('black')
        elif i[0] == 1:
            labelColor.append('green')
        else:
            labelColor.append('red')
    # 这里只演示属性1与2对比，其余两个图不做演示
    axs[0][0].scatter(x=data[:, 0], y=data[:, 1], color=labelColor, s=15, alpha=0.5)
    axs0_title = axs[0][0].set_title("属性1与属性2")
    axs0_xlabel = axs[0][0].set_xlabel("属性1")
    axs0_ylabel = axs[0][0].set_ylabel("属性2")
    plt.setp(axs0_title, size=9, weight='bold', color='red')
    plt.setp(axs0_xlabel, size=7, weight='bold', color='black')
    plt.setp(axs0_ylabel, size=7, weight='bold', color='black')
    # 设置图例
    hate = mlines.Line2D([], [], color='black', marker='.', markersize=6, label='讨厌')
    like = mlines.Line2D([], [], color='green', marker='.', markersize=6, label='喜欢')
    love = mlines.Line2D([], [], color='red', marker='.', markersize=6, label='酷爱')
    # 添加图例
    axs[0][0].legend(handles=[hate, like, love])
    plt.show()


def autoNorm(dataSet):
    '''
    三者的地位是同等的，然而数值的不同范围影响很大，归一是必须的
    :param dataSet: 归一化之前的数据集
    :return: 归一化后的数据集
    '''
    # 获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    # 最大值和最小值的范围
    ranges = maxVals - minVals
    # shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    # 返回dataSet的行数
    m = dataSet.shape[0]
    # 原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    # 除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    # 返回归一化数据结果,数据范围,最小值
    return normDataSet, ranges, minVals


if __name__ == '__main__':
    filename = "dataSet.xlsx"
    data, label = parseFile(filename)
    showDatas(data, label)
    data_end, ranges, minvalue = autoNorm(data)
    testSet = np.array([[0.20085896, 0.18888045, 0.19602906], [10, 5, 7]])
    rst = classification(testSet, data_end, label, 1)
    print(data_end)
    for i in range(len(rst)):
        if rst[i][0] == 0:
            print("你可能不喜欢")
        elif rst[i][0] == 1:
            print("你可能喜欢")
        else:
            print("你可能酷爱")
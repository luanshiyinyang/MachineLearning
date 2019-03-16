# -*-coding:utf-8-*-
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot


def get_data():
    """
    读取数据，处理为pandas.Series类型
    :return:
    """
    with open('./data/data.txt') as f:
        data = f.readline()
    data = list(map(int, data.split(",")))
    data = np.array(data, dtype=np.float)
    data = pd.Series(data)
    return data


def draw_plot(data):
    """
    对数据进行绘图，观测是否是平稳时间序列
    :param data:
    :return:
    """
    data.index = pd.Index(sm.tsa.datetools.dates_from_range('1927', '2016'))
    data.plot(figsize=(12, 8))
    plt.show()


def diff_data(data):
    """
    选择合适的p,q，以求使用ARIMA(p,d,q)模型
    :param data:
    :return:
    """
    # 一阶差分
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    diff1 = data.diff(1)
    diff1.plot(ax=ax1)
    # 二阶差分
    ax2 = fig.add_subplot(212)
    diff2 = data.diff(2)
    diff2.plot(ax=ax2)
    plt.show()


def choose_pq(data):
    """
    选择合适的p和q
    :param data:
    :return:
    """
    # 检查平稳时间序列的自相关图和偏自相关图
    diff1 = data.diff(1)
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)
    plt.show()


def choose_model(data):
    """
    获取最佳模型
    :param data:
    :return:
    """
    arma_mod70 = sm.tsa.ARMA(data, (7, 0)).fit()
    print(arma_mod70.aic, arma_mod70.bic, arma_mod70.hqic)
    arma_mod30 = sm.tsa.ARMA(data, (0, 1)).fit()
    print(arma_mod30.aic, arma_mod30.bic, arma_mod30.hqic)
    arma_mod71 = sm.tsa.ARMA(data, (7, 1)).fit()
    print(arma_mod71.aic, arma_mod71.bic, arma_mod71.hqic)
    arma_mod80 = sm.tsa.ARMA(data, (8, 0)).fit()
    print(arma_mod80.aic, arma_mod80.bic, arma_mod80.hqic)


def valid_model(data):
    """
    模型检验
    :param data:
    :return:
    """

    arma_mod80 = sm.tsa.ARMA(data, (8, 0)).fit()
    resid = arma_mod80.resid
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data, lags=40, ax=ax1)
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags=40, ax=ax2)
    plt.show()

    print(sm.stats.durbin_watson(arma_mod80.resid.values))

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    fig = qqplot(resid, line='q', ax=ax, fit=True)
    plt.show()

    r, q, p = sm.tsa.acf(resid.values.squeeze(), qstat=True)
    data = np.c_[range(1, 41), r[1:], q, p]
    table = pd.DataFrame(data, columns=['lag', 'AC', 'Q', 'Prob(>Q)'])
    print(table.set_index('lag'))


def predict(data):
    """
    模型预测
    :param data:
    :return:
    """
    data.index = pd.Index(sm.tsa.datetools.dates_from_range('1927', '2016'))
    arma_mod80 = sm.tsa.ARMA(data, (8, 0)).fit()
    predict_sunspots = arma_mod80.predict('2016', '2026', dynamic=True)
    print(predict_sunspots)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = data.ix['1927':].plot(ax=ax)
    fig = arma_mod80.plot_predict('2016', '2026', dynamic=True, ax=ax, plot_insample=False)
    plt.show()


if __name__ == '__main__':
    data = get_data()
    # draw_plot(data)
    # diff_data(data)
    # choose_pq(data)
    # choose_model(data)
    # valid_model(data)
    predict(data)

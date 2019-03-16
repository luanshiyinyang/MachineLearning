# 时间序列分析预测法
- 简述
	- 在之前，写了不少关于分类的算法，其中有传统机器学习算法如KNN、SVM，也有深度学习领域的算法如多层感知机，但是可以发现这里的算法核心思路都没有变化，利用一部分已有标签的数据训练模型，让模型去处理没有标签的数据。其实这里的分类只是分类的一种叫做有监督分类（有给定的标准，就是训练集），还有一种分类叫做无监督分类（没有标准），只是将特征接近的归为一类，又称为聚类问题。聚类的问题稍有复杂会在后面提到，这里会提及数据挖掘领域另一个大的问题方向---**预测**。
	- 预测是人们根据事物的发展规律、历史和现状，分析影响其变化的因素，对其发展前景和趋势的一种推测。预测的方法和形式多种多样，根据方法本身的性质特点将预测方法分为定性预测方法、时间序列分析、因果关系预测。
	- 时间序列分析预测法是一种定性分析方法，它是在时间序列变量分析的基础上，运用一定的数学方法建立预测模型，使时间趋势向外延伸，从而预测市场的发展变化趋势，确定变量预测值，也称为时间序列分析法、历史延伸法和外推法。
		- 确定性时间序列分析预测法
			- 这种预测方法使用的数学模型是不考虑随机项的非统计模型，是利用反映事物具有确定性的时间序列进行预测的方法，包括平均法、指数平滑法、趋势外推法、季节指数预测法等。
		- 随机性时间序列分析预测法
			- 这种方法是利用反映事物具有随机性的时间序列进行预测的方法。它的基本思想是假定预测对象是一个随机时间序列，然后利用统计数据估计该随机过程的模型，根据最终的模型做出最佳的预测。由于这种方法考虑的因素比较多，计算过程复杂，计算量大，因此发展缓慢。一般市场预测使用的是**确定性分析预测法**。
- 原理
	- 一般，时间序列分析通常将各种可能发生作用的因素进行分类，传统的分类方法是按各种因素的特点或影响效果分为四大类：长期趋势（T)，季节变动（S），循环变动（C）和不规则变动（I）。
	- 时间序列是指同一变量按时间发生的先后顺序排列起来的一组观察值或者记录值。时间序列分析预测法依据的是惯性原理，所以它建立在某经济变量过去的发展变化趋势的基础上，也就是该经济变量未来的发展变化趋势是假设的。然而从事物发展变化的规律来看，同一经济变量的发展趋势在不同时期是不可能完全相同的。这样只有将定性预测和时间序列分析预测结合在一起，才能收到最佳效果。即首先通过定性预测，在保证惯性原理成立的前提下，再运用时间序列分析预测法进行定量预测。
	- 步骤
		- 收集历史资料，加以整理，编成时间序列，并根据时间序列绘成统计图。
		- 分析时间序列。时间序列中的每一时期的数值都是由许许多多不同的因素同时发生作用后的综合结果。
		- 求时间序列的长期趋势、季节变动和不规则变动的值，并选定近似的数学模式来代表它们。对于数学模式中的未知参数，使用合适的技术方法求出其值。
		- 利用时间序列资料求出长期趋势、季节变动和不规则变动的数学模型后，就可以利用它来预测未来的长期趋势值T和季节变动值S，在可能的情况下预测不规则变动值I。然后使用一下模式计算出未来的时间序列预测值Y：
			- 加法模式：T+S+I=Y
			- 乘法模式：T*S*I=Y
		- 如果不规则变动的预测值难以求解，就只求出长期趋势和季节变动的预测值，以两者的和或者积作为时间序列预测值。如果经济现象本身没有季节变动或者不需要预测分季节，分月度的情况，则长期趋势的值就是时间序列的预测值，即T=Y。但是注意这个预测值只反映未来的发展趋势，即使很准确的趋势线也只是一个平均作用，实际值将围绕其上下波动。
- 特点
	- 撇开了事物发展的因果关系去分析事物过去和未来的联系。
	- 假设过去的趋势会延伸到未来。
	- 时间序列数据变动存在规律性和不规律性。
- 常用预测法
	- 指数平滑法
	- 季节性趋势预测法
	- 市场寿命周期预测法
- 实战
	- 根据一年的历史数据预测后10年数据趋势
	- 使用ARIMA（p，d，q）模型
	- 流程
		- 读取数据
		- 对数据绘图，观察是否为平稳序列
			- ![](https://img-blog.csdnimg.cn/201903161048174.png)
		- 对非平稳序列进行n阶差分
			- ![](https://img-blog.csdnimg.cn/20190316105703690.png)
			- 可以看到，一阶差分已经平稳，二阶变动不大，可以选择d=1。
		- 选择合适的p，q
			- ![](https://img-blog.csdnimg.cn/20190316110318228.png)
			- 如何根据相关图选取ARIMA模型，这里不多提及了。
		- 确定模型
			- 选定AIC、BIC、HQIC均值最小的ARMA(8,0)
		- 模型检验
			- 合理
		- 模型预测
			- ![](https://img-blog.csdnimg.cn/20190316111755573.png)
			- 预测结果还是比较合理的
	- 代码
		- ```python
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
			
			```
- 补充说明
	- 参考书《Python3数据分析与机器学习实战》
	- 具体数据集和代码可以查看我的GitHub,欢迎star或者fork
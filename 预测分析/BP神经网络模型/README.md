# BP神经网络模型
- 简介
	- BP网络（Back-Propagation Network）是1986年被提出的，是一种按误差逆向传播算法训练的多层前馈网络，是目前应用最广泛的神经网络模型之一，用于函数逼近、模型识别分类、数据压缩和时间序列预测等。
	- ![来源网络](https://img-blog.csdnimg.cn/20190317151418863.png)
	- BP网络又称为反向传播神经网络，它是一种有监督的学习算法，具有很强的自适应、自学习、非线性映射能力，能较好地解决数据少、信息贫、不确定性问题，且不受非线性模型的限制。一个典型的BP网络应该包括三层:输入层、隐含层和输出层。（关于神经网络的基础术语等知识可以查看[我之前的博客](https://blog.csdn.net/zhouchen1998/article/details/88081151)）各层之间全连接，同层之间无连接。隐含层可以有很多层，对于一般的神经网络而言，单层的隐含层已经足够了。上图是一个典型的BP神经网络结构图。
	- 学习过程
		- 正向传播
			- 输入信号从输入层经过各个隐含层向输出层传播从，在输出层得到实际的响应值，若实际值与期望值误差较大，就会转入误差反向传播阶段。
		- 反向传播
			- 按照梯度下降的方法从输出层经过各个隐含层并逐层不断地调整各神经元的连接权值和阈值，反复迭代，直到网络输出的误差减少到可以接受的程度，或者进行到预先设定的学习次数。
	- BP神经网络通过有指导的学习方式进行训练和学习。标准的BP算法采用误差函数按梯度下降的方法学习，使网络的设计输出值和期望输出值之间的均方误差最小。BP神经网络的传输函数通常采用sigmoid函数，而输出层则采用线性传输函数。
- 原理
	- 输入输出层的设计
		- 输入层
			- 输入层各神经元负责接收来自外界的输入信息，并传递给中间层各神经元，它的节点数为输入变量的个数。
		- 输出层
			- 输出层向外界输出信息处理结果。它的节点个数为输出变量的个数。
	- 隐含层设计
		- 内部信息处理层，负责信息变换，根据信息变换能力的需求，中间层可以设计为单隐层或多隐层结构；最后一个隐含层传递信息到输出层各神经元，经过进一步处理后，完成一次学习的正向传播处理过程。
		- 有关研究表明，一个隐含层的神经网络，只要隐节点足够多，就可以以任意精度逼近一个非线性函数。因此，通常采用含有一个隐层的三层多输入单输出的BP神经网络建立预测模型。
		- 在网络设计过程中，隐层神经元数的确定十分重要。隐层神经元数目过多，会加大网络计算量并容易产生过度拟合问题；神经元数目过少，则会影响网络性能，达不到预期效果。网络中隐层神经元的数目与实际问题的复杂程度、输入和输出层的神经元数及对期望误差的设定有着直接的关系。目前，对于隐层的神经元数目的确定没有明确的公式，只有一些基于经验的公式，神经元的个数最终需要根据经验和多次试验确定。
			- $L= \sqrt {n+m} +  a$
			- 其中，n为输入层神经元个数，m为输出层神经元个数，a为1到10之间的常数。
	- BP算法改进
		- 虽然BP神经网络具有高度非线性和较强的泛化能力，但也存在收敛速度慢、迭代步数多、易陷入局部极小和全局搜索能力差等缺点。可以采用增加动量项、自适应调节学习率、引入陡度因子等方法进行改进。
			- 增加动量项
				- 加速算法收敛
				- $w_{ij} = w_{ij} - \eta_1 \times \delta_{ij}\times x_i + \alpha ∆w_{ij}$
				- 其中，动量因子$\alpha$一般为0.1~0.8。
			- 自适应调节率
			- 引入陡度因子
		- 通常BP神经网络在训练之前会将数据归一化，映射到更小的区间内。
- 实战
	- 使用神经网络预测公路运量
	- 过程
		- 划分训练集和验证集
		- 建立模型，设定模型参数
		- 对模型进行训练并可视化损失函数变化
			- ![](https://img-blog.csdnimg.cn/20190317162247143.png)
		- 模型验证并使用（观察预测值和真实值对比）
			- ![](https://img-blog.csdnimg.cn/20190317162645796.png)
			- 可以看到，效果还是很显著的
	- ```python
		# -*-coding:utf-8-*-
		import numpy as np
		import matplotlib.pyplot as plt
		
		
		def logsig(x):
		    """
		    定义激活函数
		    :param x:
		    :return:
		    """
		    return 1/(1+np.exp(-x))
		
		
		def get_Data():
		    """
		    读入数据，转为归一化矩阵
		    :return:
		    """
		    # 读入数据
		    # 人数(单位：万人)
		    population = [20.55, 22.44, 25.37, 27.13, 29.45, 30.10, 30.96, 34.06, 36.42, 38.09, 39.13, 39.99, 41.93, 44.59,
		                  47.30, 52.89, 55.73, 56.76, 59.17, 60.63]
		    # 机动车数(单位：万辆)
		    vehicle = [0.6, 0.75, 0.85, 0.9, 1.05, 1.35, 1.45, 1.6, 1.7, 1.85, 2.15, 2.2, 2.25, 2.35, 2.5, 2.6, 2.7, 2.85, 2.95,
		               3.1]
		    # 公路面积(单位：万平方公里)
		    roadarea = [0.09, 0.11, 0.11, 0.14, 0.20, 0.23, 0.23, 0.32, 0.32, 0.34, 0.36, 0.36, 0.38, 0.49, 0.56, 0.59, 0.59,
		                0.67, 0.69, 0.79]
		    # 公路客运量(单位：万人)
		    passengertraffic = [5126, 6217, 7730, 9145, 10460, 11387, 12353, 15750, 18304, 19836, 21024, 19490, 20433, 22598,
		                        25107, 33442, 36836, 40548, 42927, 43462]
		    # 公路货运量(单位：万吨)
		    freighttraffic = [1237, 1379, 1385, 1399, 1663, 1714, 1834, 4322, 8132, 8936, 11099, 11203, 10524, 11115, 13320,
		                      16762, 18673, 20724, 20803, 21804]
		
		    # 将数据转换成矩阵，并使用最大最小归一数据
		    # 输入数据
		    samplein = np.mat([population, vehicle, roadarea])  # 3*20
		    # 得到最大最小值，方便归一
		    sampleinminmax = np.array([samplein.min(axis=1).T.tolist()[0], samplein.max(axis=1).T.tolist()[0]]).transpose()
		    # 输出数据
		    sampleout = np.mat([passengertraffic, freighttraffic])  # 2*20
		    # 得到最大最小值，方便归一
		    sampleoutminmax = np.array([sampleout.min(axis=1).T.tolist()[0], sampleout.max(axis=1).T.tolist()[0]]).transpose()
		    sampleinnorm = (2 * (np.array(samplein.T) - sampleinminmax.transpose()[0]) / (
		                sampleinminmax.transpose()[1] - sampleinminmax.transpose()[0]) - 1).transpose()
		    sampleoutnorm = (2 * (np.array(sampleout.T).astype(float) - sampleoutminmax.transpose()[0]) / (
		                sampleoutminmax.transpose()[1] - sampleoutminmax.transpose()[0]) - 1).transpose()
		
		    # 给输入样本添加噪声
		    noise = 0.03 * np.random.rand(sampleoutnorm.shape[0], sampleoutnorm.shape[1])
		    sampleoutnorm += noise
		    return samplein, sampleout, sampleinminmax, sampleoutminmax, sampleinnorm, sampleoutnorm
		
		
		def model_create():
		    """
		    建立模型并训练
		    :return:
		    """
		    maxepochs = 60000
		    learnrate = 0.035
		    errorfinal = 0.65 * 10 ** (-3)
		    samnum = 20
		    indim = 3
		    outdim = 2
		    hiddenunitnum = 8
		    w1 = 0.5 * np.random.rand(hiddenunitnum, indim) - 0.1
		    b1 = 0.5 * np.random.rand(hiddenunitnum, 1) - 0.1
		    w2 = 0.5 * np.random.rand(outdim, hiddenunitnum) - 0.1
		    b2 = 0.5 * np.random.rand(outdim, 1) - 0.1
		
		    errhistory = []
		    # 开始训练模型
		    samplein, sampleout, sampleinminmax, sampleoutminmax, sampleinnorm, sampleoutnorm = get_Data()
		    for i in range(maxepochs):
		        hiddenout = logsig((np.dot(w1, sampleinnorm).transpose() + b1.transpose())).transpose()
		        networkout = (np.dot(w2, hiddenout).transpose() + b2.transpose()).transpose()
		        err = sampleoutnorm - networkout
		        sse = sum(sum(err ** 2))
		        errhistory.append(sse)
		        if sse < errorfinal:
		            break
		        delta2 = err
		        delta1 = np.dot(w2.transpose(), delta2) * hiddenout * (1 - hiddenout)
		        dw2 = np.dot(delta2, hiddenout.transpose())
		        db2 = np.dot(delta2, np.ones((samnum, 1)))
		        dw1 = np.dot(delta1, sampleinnorm.transpose())
		        db1 = np.dot(delta1, np.ones((samnum, 1)))
		        w2 += learnrate * dw2
		        b2 += learnrate * db2
		        w1 += learnrate * dw1
		        b1 += learnrate * db1
		
		    # 绘制误差曲线图
		    errhistory10 = np.log10(errhistory)
		    minerr = min(errhistory10)
		    plt.plot(errhistory10)
		    plt.plot(range(0, i + 1000, 1000), [minerr] * len(range(0, i + 1000, 1000)))
		    ax = plt.gca()
		    ax.set_yticks([-2, -1, 0, 1, 2, minerr])
		    ax.set_yticklabels([u'$10^{-2}$', u'$10^{-1}$', u'$10^{1}$', u'$10^{2}$', str(('%.4f' % np.power(10, minerr)))])
		    ax.set_xlabel('iteration')
		    ax.set_ylabel('error')
		    ax.set_title('Error Histroy')
		    plt.savefig('errorhistory.png', dpi=700)
		    plt.close()
		
		    # 实现仿真输出和实际输出对比图
		    hiddenout = logsig((np.dot(w1, sampleinnorm).transpose() + b1.transpose())).transpose()
		    networkout = (np.dot(w2, hiddenout).transpose() + b2.transpose()).transpose()
		    diff = sampleoutminmax[:, 1] - sampleoutminmax[:, 0]
		    networkout2 = (networkout + 1) / 2
		    networkout2[0] = networkout2[0] * diff[0] + sampleoutminmax[0][0]
		    networkout2[1] = networkout2[1] * diff[1] + sampleoutminmax[1][0]
		
		    sampleout = np.array(sampleout)
		
		    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
		    line1, = axes[0].plot(networkout2[0], 'k', marker=u'$\circ$')
		    line2, = axes[0].plot(sampleout[0], 'r', markeredgecolor='b', marker=u'$\star$', markersize=9)
		
		    axes[0].legend((line1, line2), ('simulation output', 'real output'), loc='upper left')
		
		    yticks = [0, 20000, 40000, 60000]
		    ytickslabel = [u'$0$', u'$2$', u'$4$', u'$6$']
		    axes[0].set_yticks(yticks)
		    axes[0].set_yticklabels(ytickslabel)
		    axes[0].set_ylabel(u'passenger traffic$(10^4)$')
		
		    xticks = range(0, 20, 2)
		    xtickslabel = range(1990, 2010, 2)
		    axes[0].set_xticks(xticks)
		    axes[0].set_xticklabels(xtickslabel)
		    axes[0].set_xlabel(u'year')
		    axes[0].set_title('Passenger Traffic Simulation')
		
		    line3, = axes[1].plot(networkout2[1], 'k', marker=u'$\circ$')
		    line4, = axes[1].plot(sampleout[1], 'r', markeredgecolor='b', marker=u'$\star$', markersize=9)
		    axes[1].legend((line3, line4), ('simulation output', 'real output'), loc='upper left')
		    yticks = [0, 10000, 20000, 30000]
		    ytickslabel = [u'$0$', u'$1$', u'$2$', u'$3$']
		    axes[1].set_yticks(yticks)
		    axes[1].set_yticklabels(ytickslabel)
		    axes[1].set_ylabel(u'freight traffic$(10^4)$')
		
		    xticks = range(0, 20, 2)
		    xtickslabel = range(1990, 2010, 2)
		    axes[1].set_xticks(xticks)
		    axes[1].set_xticklabels(xtickslabel)
		    axes[1].set_xlabel(u'year')
		    axes[1].set_title('Freight Traffic Simulation')
		
		    fig.savefig('simulation.png', dpi=500, bbox_inches='tight')
		    plt.show()
		
		
		if __name__ == '__main__':
		    model_create()
		```
- 补充说明
	- 参考书《Python3数据分析与机器学习实战》
	- 具体数据集和代码可以查看我的GitHub,欢迎star或者fork
# Boosting和AdaBoost
- 简介
	- Bagging采用的是一种多个分类器简单评分的方式。而Boosting是和Bagging对应的一种将弱分类器组合成为强分类器的算法框架，它根据分类器学习误差率来更新训练样本的权重。AdaBoost算法就是Boosting算法的一种。它建立在多个若分类器的基础上，为分类器进行权重赋值，性能好的分类器能获得更多权重，从而使评分效果更理想。
- 原理
	- AdaBoost算法的基本步骤有以下三步。
		- 初始化样本权重，一般进行等权重处理。
		- 训练弱分类器，根据每个分类器的结果更新权重，再进行训练，直到符合条件。
		- 将弱分类器集合成为强分类器，一般是分类器误差小的权重大。
	- 以二元分类问题为例分析AdaBoost算法的原理，对于多元问题和回归问题可以进行类似推理。
		- 假设训练集样本是：$$ T=\{(x_1,y_1),(x_2,y_2),...,(x_m,y_m)\} $$ 
		- 训练集第k个弱学习器的输出权重为：$$ D(k)=(w_{k1},w_{k2},...w_{kn}); w_{1i}=1/m;i=1,2,...m $$ 
		- 假设二元分类问题的输出为$\{-1,1\}$，则第k个弱分类器$G_k(x)$在训练集上的加权误差率为$$ e_k=p(G_k(x_i) \not= y_i)=\sum_{i=1}^m W_{ki}|(G_k(x_i) \not= y_i) $$
		- 第k个弱分类器$G_k(x)$的权重系数为：$$ \alpha_k={1 \over 2}\log {1-e_k \over e_k} $$
		- 从上式可以看出，如果分类误差率$e_k$越大，则对应的弱分类器权重系数$\alpha_k$越小。也就是说，误差率小的弱分类器权重系数越大。
		- 样本权重的更新过程如下，假设第k个弱分类器的样本集权重系数为$D(k)=(w_{k1},w_{k2},...w_{km})$，则对应的第k+1个弱分类器的样本集权重系数为：$$ w_{k+1,i}={w_{ki} \over Z_K}exp(-\alpha_ky_iG_k(x_i)) $$，这里$Z_K$是规范因子。
		- 从上面的$w_{k+1,i}$的计算公式可以看出，如果第i个样本分类错误，则$y_iG_k(x_i)$导致样本的权重在第k+1个弱分类器中增大；如果分类正确，则权重在第k+1个弱分类器中减小。
		- AdaBoost算法采用加权平均方法进行融合，最终的强分类器为：$$ f(x)=sign(\sum_{k=1}^k \alpha_kG_k(x)) $$
	- AdaBoost具有原理简单、分类精度高、能使用各种分类模型来构建弱学习器、不容易过拟合等特点，在实际中得到了广泛应用。
- 实战
	- 使用AdaBoost进行二元分类
	- 分类器使用sklearn模型（默认使用CART决策树作为弱分类器）
	- 数据集随机生成
	- 代码
		- ```python
			# -*-coding:utf-8-*-
			import numpy as np
			import matplotlib.pyplot as plt
			from sklearn.ensemble import AdaBoostClassifier
			from sklearn.tree import DecisionTreeClassifier
			from sklearn.datasets import make_gaussian_quantiles
			
			# 生成随机数据集
			X1, y1 = make_gaussian_quantiles(
			    cov=2.0,
			    n_samples=500,
			    n_features=2,
			    n_classes=2,
			    random_state=1
			)
			X2, y2 = make_gaussian_quantiles(
			    mean=(3, 3),
			    cov=1.5,
			    n_samples=400,
			    n_features=2,
			    n_classes=2,
			    random_state=1
			)
			X = np.concatenate((X1, X2))
			y = np.concatenate((y1, -y2+1))
			
			# 绘制散点图，将数据可视化
			plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
			plt.show()
			
			# 该参数准确率最高
			bdt = AdaBoostClassifier(
			    DecisionTreeClassifier(max_depth=2, min_samples_split=20, min_samples_leaf=5),
			    algorithm='SAMME',
			    n_estimators=600,
			    learning_rate=0.7
			)
			
			# 训练
			bdt.fit(X, y)
			# 生成网格图查看拟合区域
			x_min, x_max = X[:, 0].min()-1, X[:, 0].max()+1
			y_min, y_max = X[:, 1].min()-1, X[:, 1].max()+1
			xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
			Z = bdt.predict(np.c_[xx.ravel(), yy.ravel()])
			Z = Z.reshape(xx.shape)
			plt.contour(xx, yy, Z, cmap=plt.cm.Paired)
			plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
			plt.show()
			
			# 查看AdaBoost方法的分类精度
			print(bdt.score(X, y))
			```
	- 运行效果
		- ![](https://img-blog.csdnimg.cn/20190401193109277.png)
- 补充说明
	- 参考书为《Python3数据分析与机器学习实战》，对部分错误修改
	- 具体数据集和代码见我的Github，欢迎Star或者Fork
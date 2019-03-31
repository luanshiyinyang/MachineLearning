# Bagging和随机森林
- 前言
	- 集成学习是目前机器学习的一大热门方向。简单来说，集成学习就是组合许多弱模型以得到一个预测结果比较好的强模型。对于常见的分类问题就是指采用多个分类器对数据集进行预测，把这些分类器的分类结果进行某种组合（如投票）决定分类结果，从而整体提高分类器的泛化能力。
	- 集成学习对于**大数据集和不充分数据**都有很好的效果。因为一些简单模型数据量太大而很难训练，或者只能学习到一部分，而集成学习方法可以有策略地将数据集划分成一些小数据集，并分别进行训练，之后根据一些策略进行组合。相反，如果数据量很少，可以使用bootstrap进行抽样，得到多个数据集，分别进行训练后再组合。
	- 集成学习中组合的模型可以是同一类型的模型，也可以是不同类型的模型。根据采用的数据采样、预测方法等的不同，常见的集合组合策略主要有平均算法和Boosting两类。其中，平均算法利用不同估计算法的结果平均进行预测，在估计模型上按照不同的变化形式可以进一步划分为粘合（Pasting）、分袋（Bagging）、子空间（Subspacing）和分片（Patches）等。Boosting算法通过一系列聚合的估计模型加权平均进行预测。
	- 其中比较典型的算法就是随机森林方法和AdaBoost方法。
- 简介
	- 随机森林算法是一种典型的基于决策树的集成算法。它是通过集成学习的思想将多棵树集成的一种算法。20世纪80年代Breiman等发明分类树的算法，通过反复二分数据进行分类或回归，使机器学习模型较传统的神经网络方法计算量大大降低。2001年Breiman把分类树组合成随机森林，即在变量和数据的使用上进行随机化，生成许多分类树，再汇总分类树的结果。随机森林在运算量没有显著提高的前提下提高了预测精度，它对多元非线性不敏感。结果对缺失数据和非平衡的数据比较稳健，可以起到很好地预测多达几千个解释变量的作用。
- 原理
	- 传统的Bagging抽样方法，会从数据集中重复抽取大小为N的子样本，这就导致有的数据会重复出现。抽取子样本后，使用原始数据集作为测试集，而多个子样本作为训练集。与Bagging方法相比，随机森林方法首先从样本中随机抽取n个样本，然后结合随机选择的特征K，对它们进行m次决策树构建，这里多了一次针对特征的随机选择过程。
	- 随机森林的每一棵分类树为二叉树，其生成遵循自顶向下的递归分裂原则，即从根节点开始依次对训练集进行划分。在二叉树中，根节点包含全部训练数据，按照节点纯度最小原则，分裂为左节点和右节点，它们分别包含训练数据的一个子集。按照同样的规则节点继续分裂，直到满足分支停止规则而停止生长。若节点n上的分类数据全部来自同一类别，则此节点的纯度I(n)=0，纯度度量方法采用Gini准则。
	- 具体实现过程如下。
		- 原始训练集为N，采用Bootstrap法有放回地随机抽取k个新的自助样本集，并由此构建k棵分类树，每次未被抽到的样本组成了k个袋外数据。
		- 设有M个变量，则在每一棵树的每个节点处随机抽取M1个变量，然后在M1中选择一个最具有分类能力的变量，变量分类的阈值通过检查每一个分类点确定。
		- 每一棵树最大限度地生长，不做任何修剪。
		- 将生成的多棵分类树组成随机森林，用随机森林分类器对新的数据进行判别与分类，分类结果按树分类器的投票多少而定。
	- 随机森林是一种利用多个分类树对数据进行判别与分类的方法，其特点主要表现在数据随机选取和特征随机选取两个方面。
		- 数据随机选取是指从原始数据集中选取数据组成不同的子数据集，利用这些子数据集构建子决策树，观察子决策树的分类结果，随机森林的分类结果属于子决策树的分类结果指向多的那个。
		- 特征随机选取是指随机森林中子树的每一个分裂过程并未用到所有的待选特征，而是从所有的待选特征中随机选取一定的特征，之后再在随机选取的特征中选取最优的特征。
- 实战
	- 使用随机森林预测乘客存活概率
	- 使用Kaggle提供的Titanic数据集
	- 这道竞赛题，训练数据少，使用逻辑回归、SVM等算法，容易出现预测糟糕的情况。可以采用分析特征重要性的同时，建立随机森林模型来进行分类。
	- 基于sklearn实现。
	- 代码
		- ```python
			# -*-coding:utf-8-*-
			import numpy as np
			import pandas as pd
			from sklearn.ensemble import RandomForestClassifier
			from sklearn.model_selection import cross_val_score
			
			
			def get_data():
			    train = pd.read_csv('data/train.csv', dtype={'Age': np.float64})
			    test = pd.read_csv('data/test.csv', dtype={'Age': np.float64})
			    return train, test
			
			
			def harmonize_data(titanic):
			    """
			    预处理数据，随机森林不允许非数值、空置等
			    :param titanic:
			    :return:
			    """
			    titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())
			    titanic.loc[titanic['Sex']=='male','Sex']=0
			    titanic.loc[titanic['Sex']=='female','Sex']=1
			    titanic['Embarked'] = titanic['Embarked'].fillna('S')
			    titanic.loc[titanic['Embarked']=='S','Embarked']=0
			    titanic.loc[titanic['Embarked']=='C','Embarked']=1
			    titanic.loc[titanic['Embarked']=='Q','Embarked']=2
			    titanic['Fare'] = titanic['Fare'].fillna(titanic['Fare'].median())
			    return titanic
			
			
			def create_submission(alg, train, test, predictors, filename):
			    """
			    文件输出，一般竞赛平台要求提交数据集为id+label
			    :param alg:
			    :param train:
			    :param test:
			    :param predictors:
			    :param filename:
			    :return:
			    """
			    alg.fit(train[predictors], train['Survived'])
			    predictions = alg.predict(test[predictors])
			    submission = pd.DataFrame({
			        'PassengerId': test['PassengerId'],
			        'Survived': predictions
			    })
			    submission.to_csv(filename, index=False)
			
			
			if __name__ == '__main__':
			    train, test = get_data()
			    train_data = harmonize_data(train)
			    test_data = harmonize_data(test)
			    # 确定模型的特征
			    predictors = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
			    # 建模
			    alg = RandomForestClassifier(
			        random_state=1,
			        n_estimators=150,
			        min_samples_split=4,
			        min_samples_leaf=2
			    )
			    # 交叉验证
			    scores = cross_val_score(
			        alg,
			        train_data[predictors],
			        train_data['Survived'],
			        cv=3
			    )
			    print(scores.mean())
			    print(scores.std())
			    # 预测结果输出
			    create_submission(alg, train_data, test_data, predictors, 'data/result.csv')
			```
	- 效果
		- ![](https://img-blog.csdnimg.cn/20190331203730347.png)
- 补充说明
	- 参考书为《Python3数据分析与机器学习实战》，对部分错误修改
	- 具体数据集和代码见我的Github，欢迎Star或者Fork
	- Kaggle是个不错的数据挖掘竞赛平台
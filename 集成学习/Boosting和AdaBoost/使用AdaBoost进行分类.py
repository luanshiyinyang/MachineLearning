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
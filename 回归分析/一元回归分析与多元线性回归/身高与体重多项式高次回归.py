# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# 输入训练数据和测试数据
X_train = [[0.86], [0.96], [1.12], [1.35], [1.55], [1.63], [1.71], [1.78]]
y_train = [[12], [15], [20], [35], [48], [51], [59], [66]]
X_test = [[0.75], [1.08], [1.26], [1.51], [1.6], [1.85]]
y_test = [[10], [17], [27], [41], [50], [75]]

k_range = range(2, 10)
k_scores = []

regressor = LinearRegression()
regressor.fit(X_train, y_train)
k_scores.append(regressor.score(X_test, y_test))

for k in k_range:
    k_featurizer = PolynomialFeatures(degree=k)
    X_train_k = k_featurizer.fit_transform(X_train)
    X_test_k = k_featurizer.transform(X_test)
    regressor_k = LinearRegression()
    regressor_k.fit(X_train_k, y_train)
    k_scores.append(regressor_k.score(X_test_k, y_test))

for i in range(0, 8):
    print('%d项式r^2是%.2f' % (i + 1, k_scores[i]))

plt.plot([1, 2, 3, 4, 5, 6, 7, 8, 9], k_scores)
plt.show()

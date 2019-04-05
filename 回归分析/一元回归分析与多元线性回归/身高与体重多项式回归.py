# -*-coding:utf-8-*-
# 导入模块
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontProperties
from sklearn.preprocessing import PolynomialFeatures

# 字体
font = FontProperties(fname=r'c:\Windows\Fonts\msyh.ttc', size=15)


# 显示数据
def runplt():
    plt.figure()
    plt.title('身高与体重一元关系', fontproperties=font)
    plt.xlabel('身高(米)', fontproperties=font)
    plt.ylabel('体重(公斤)', fontproperties=font)
    plt.axis([0.5, 2, 5, 85], fontproperties=font)
    plt.grid(True)
    return plt


# 输入训练数据和测试数据
X_train = [[0.86], [0.96], [1.12], [1.35], [1.55], [1.63], [1.71], [1.78]]
y_train = [[12], [15], [20], [35], [48], [51], [59], [66]]
X_test = [[0.75], [1.08], [1.26], [1.51], [1.6], [1.85]]
y_test = [[10], [17], [27], [41], [50], [75]]

# 显示
plt = runplt()
regressor = LinearRegression()
regressor.fit(X_train, y_train)
xx = np.linspace(0, 26, 100)
yy = regressor.predict(xx.reshape(xx.shape[0], 1))
plt.plot(X_train, y_train, 'k.')
plt.plot(xx, yy)

# 转换器
quadratic_fearurizer = PolynomialFeatures(degree=2)
X_train_quadratic = quadratic_fearurizer.fit_transform(X_train)
X_test_quadratic = quadratic_fearurizer.transform(X_test)
regressor_quadratic = LinearRegression()
regressor_quadratic.fit(X_train_quadratic, y_train)
xx_quadratic = quadratic_fearurizer.transform(xx.reshape(xx.shape[0], 1))
plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')
plt.show()
print('一元线性回归r^2:%.2f' % regressor.score(X_test, y_test))
print('二元线性回归r^2:%.2f' % regressor_quadratic.score(X_test_quadratic, y_test))

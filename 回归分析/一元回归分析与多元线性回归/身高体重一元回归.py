# -*-coding:utf-8-*-
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from sklearn.linear_model import LinearRegression

# 字体
font = FontProperties(fname=r'c:\Windows\Fonts\msyh.ttc', size=15)


# 显示数据
def runplt():
    plt.figure()
    plt.title('身高与体重一元关系', fontproperties=font)
    plt.xlabel('身高(米)', fontproperties=font)
    plt.ylabel('体重(公斤)', fontproperties=font)
    plt.axis([0, 2, 0, 85], fontproperties=font)
    plt.grid(True)
    return plt


# 输入训练数据
X = [[0.86], [0.96], [1.12], [1.35], [1.55], [1.63], [1.71], [1.78]]
y = [[12], [15], [20], [35], [48], [51], [59], [66]]
plt = runplt()
plt.plot(X, y, 'k.')
plt.show()

# 创建模型
model = LinearRegression()
# 将训练数据放入模型中
model.fit(X, y)
# 预测
print('预测身高为1.67米的体重是：%2.f公斤' % model.predict(np.array([1.67]).reshape(-1, 1)))

# 使用测试数据真题对该模型进行预测
X2 = [[0.75], [1.08], [1.26], [1.51], [1.6], [1.85]]
y2 = model.predict(X2)
plt.plot(X, y, 'k.')
plt.plot(X2, y2, 'g-')
# 残差预测值
yr = model.predict(X)
for idx, x in enumerate(X):
    plt.plot([x, x], [y[idx], yr[idx]], 'r-')
plt.show()

# R方
X_test = [[0.75], [1.08], [1.26], [1.51], [1.6], [1.85]]
y_test = [[10], [17], [27], [41], [50], [75]]
r2 = model.score(X_test, y_test)
print('R^2=%.2f' % r2)

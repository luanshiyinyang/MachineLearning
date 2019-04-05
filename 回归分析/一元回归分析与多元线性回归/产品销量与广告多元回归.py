# -*-coding:utf-8-*-
# 导入模块
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# 使用pandas读入数据
data = pd.read_csv('data/Advertising.csv')

# 转换数据
feature_cols = ['TV', 'radio', 'newspaper']
X = data[feature_cols]
y = data['sales']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.4, random_state=0)

# 循环多元回归模型
linreg = LinearRegression()
model = linreg.fit(X_train, y_train)
print(model)
print(linreg.intercept_)
print(linreg.coef_)

# 预测
y_pred = linreg.predict(X_test)
print(y_pred)

# 使用图形来对比预测数据与实际数据之间的关系
plt.figure()
plt.plot(range(len(y_pred)), y_pred, 'b', label='predict')
plt.plot(range(len(y_pred)), y_test, 'r', label='test')
plt.legend(loc='upper right')
plt.xlabel('the number of sales')
plt.ylabel('value of sales')
plt.show()

# 模型验证
sum_mean = 0
for i in range(len(y_pred)):
    sum_mean += (y_pred[i] - y_test.values[i]) ** 2
sum_erro = np.sqrt(sum_mean / 50)
print('RMSE by hand:', sum_erro)

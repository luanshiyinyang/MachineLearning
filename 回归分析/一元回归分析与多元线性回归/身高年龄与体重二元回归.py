# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from matplotlib.font_manager import FontProperties

# 准备训练数据
X = [[147, 9], [129, 7], [141, 9], [145, 11], [142, 11], [151, 13]]
y = [[34], [23], [25], [47], [26], [46]]

# 创建模型
model = LinearRegression()
# 输入数据
model.fit(X, y)

# 准备测试数据
X_test = [[149, 11], [152, 12], [140, 8], [138, 10], [132, 7], [147, 10]]
y_test = [[41], [37], [28], [27], [21], [38]]

# 预测
predictions = model.predict(X_test)

# 模型评估
print('R^2为%.2f' % model.score(X_test, y_test))
# 输出数据
for i, prediction in enumerate(predictions):
    print('Predicted:%s,Target:%s' % (prediction, y_test[i]))

# 图形显示
font = FontProperties(fname=r'c:\Windows\Fonts\msyh.ttc', size=15)
plt.title('多元回归实际值与预测值', fontproperties=font)
plt.plot(y_test, label='y_test')
plt.plot(predictions, label='predictions')
plt.legend()
plt.show()

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.metrics import auc
# X表示真实结果标签
X = np.array([1, 2, 2, 1, 2, 1, 0])
# Y表示预测结果标签
Y = np.array([2, 1, 2, 1, 2, 1, 0])
# 设定积极标签个数
fpr, tpr, thresholds = metrics.roc_curve(X, Y, pos_label=1)

# 绘制ROC曲线
plt.plot(fpr, tpr, marker='o')
plt.show()

# 求出AUC值
AUC = auc(fpr, tpr)
print(AUC)

# 得到混淆矩阵
# labels表示所有的标签种类个数
mess = metrics.confusion_matrix(X, Y, labels=[0, 1, 2])
print(mess)

score = metrics.accuracy_score(X, Y)
print(score)

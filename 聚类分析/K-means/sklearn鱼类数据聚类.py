# -*-coding:utf-8-*-
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

df = pd.read_csv('data/fish.csv')
species = list(df['species'])
del df['species']
print(df.head())
# 归一化预处理
samples = df.values
scaler = StandardScaler()
kmeans = KMeans(n_clusters=4)
pipline = make_pipeline(scaler, kmeans)
# 训练
pipline.fit(samples)
# 预测
labels = pipline.predict(samples)
# 创建DataFrame，装入鱼类数据的聚类类别号与真实种类，交叉表显示
df = pd.DataFrame({'labels': labels, 'species': species})
ct = pd.crosstab(df['labels'], df['species'])
print(ct)

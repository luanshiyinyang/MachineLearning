# 奇异值分解
- 简介
	- PCA是通过特征值分解来进行特征提取的，但它要求矩阵必须是方阵，但在实际应用场景中，经常遇到的矩阵都不是方阵，如N个学生，每个学生有M门课程，其中N!=M, 这就组成了一个M*N的非方阵矩阵，这种情况下无法使用主成分分析，也限制了特征值分解方法的使用。而奇异值分解（SVD），是线性代数中重要的一种矩阵分解，该方法对矩阵的形状没有要求。
- 原理
	- 在很多情况下，数据的一小段携带了数据集中大部分信息，而剩下的信息则要么是噪声，要么是毫不相关的信息。利用SVD实现，能够用小得多的数据集来表示原始数据集。这样做，实际上是去除了噪声和冗余数据。同样，当用户试图节省空间时，去除信息也是很有用的。
	- 假设是一个阶矩阵，则存在一个分解使得：$$ M_{m \times n} = U_{m \times n} \sum_{m \times n} V_{n \times n}^T $$
	- 式子中，为$m\times m$阶酉矩阵；$\sum$为半正定$m\times n$阶对角矩阵；而$V^T$，即V的共轭转置，是n*n阶酉矩阵。这样的分解就称为M的奇异值分解。$\sum$对角线上的元素$\sum i$，其中i即为M的奇异值。常见的做法是奇异值从大到小排列。
	- 奇异值的优点是：可以简化数据，压缩维度，去除噪声数据，提升算法结果。加快模型计算性能，可以针对任一普通矩阵进行分解（包括样本数小于特征数），不受限于方阵。
	- 奇异值的缺点是：**转化后的数据难以理解，如何与具体业务知识对应起来是难点。**
	- 那么在解决实际问题时如何知道保留多少个奇异值呢？
		- 一个典型的做法就是保留矩阵中90%的能量信息。为了计算总能量信息，将所有的奇异值求其平方和。于是可以将奇异值的平方和累加到总值的90%为止。
- 实战
	- 使用奇异值分解进行图像压缩
	- u, sigma, v = np.linalg.svd(M)
		- 其中u和v返回矩阵M的左右奇异向量，sigma返回奇异值从大到小排列的一个向量。
	- 前置知识
		- 一般彩色图像就是RGB三个图层上矩阵的叠加，每个元素值为0~255，plt可以直接读取图像，对三个图层一一处理即可。
	- SVD进行压缩步骤
		- 读取图片，分解为RGB三个矩阵。
		- 对三个矩阵分别进行SVD分解，得到对应的奇异值和奇异向量。
		- 按照一定的标准进行奇异值筛选（整体数量的一定百分比，或者奇异值和的一定百分比）。
		- 恢复矩阵。
		- 保存图像。
	- 代码
		- ```python
			# -*-coding:utf-8-*-
			import numpy as np
			from matplotlib import pyplot as plt
			
			
			def svdimage(filename, percent):
			    """
			    读取原始图像数据
			    :param filename:
			    :param percent:
			    :return:
			    """
			    original = plt.imread(filename)  # 读取图像
			    R0 = np.array(original[:, :, 0])  # 获取第一层矩阵数据
			    G0 = np.array(original[:, :, 1])  # 获取第二层矩阵数据
			    B0 = np.array(original[:, :, 2])  # 获取第三层矩阵数据
			    u0, sigma0, v0 = np.linalg.svd(R0)  # 对第一层数据进行SVD分解
			    u1, sigma1, v1 = np.linalg.svd(G0)  # 对第二层数据进行SVD分解
			    u2, sigma2, v2 = np.linalg.svd(B0)  # 对第三层数据进行SVD分解
			    R1 = np.zeros(R0.shape)
			    G1 = np.zeros(G0.shape)
			    B1 = np.zeros(B0.shape)
			    total0 = sum(sigma0)
			    total1 = sum(sigma1)
			    total2 = sum(sigma2)
			
			    # 对三层矩阵逐一分解
			    sd = 0
			    for i, sigma in enumerate(sigma0):  # 用奇异值总和的百分比来进行筛选
			        R1 += sigma * np.dot(u0[:, i].reshape(-1, 1), v0[i, :].reshape(1, -1))
			        sd += sigma
			        if sd >= percent * total0:
			            break
			
			    sd = 0
			    for i, sigma in enumerate(sigma1):  # 用奇异值总和的百分比来进行筛选
			        G1 += sigma * np.dot(u1[:, i].reshape(-1, 1), v1[i, :].reshape(1, -1))
			        sd += sigma
			        if sd >= percent * total1:
			            break
			
			    sd = 0
			    for i, sigma in enumerate(sigma2):  # 用奇异值总和的百分比来进行筛选
			        B1 += sigma * np.dot(u2[:, i].reshape(-1, 1), v2[i, :].reshape(1, -1))
			        sd += sigma
			        if sd >= percent * total2:
			            break
			
			    final = np.stack((R1, G1, B1), 2)
			    final[final > 255] = 255
			    final[final < 0] = 0
			    final = np.rint(final).astype('uint8')
			    return final
			
			
			if __name__ == '__main__':
			    filename = 'data/example.jpg'
			    for p in np.arange(.1, 1, .1):
			        print('percent is {}'.format(p))
			        after = svdimage(filename, p)
			        plt.imsave('data/' + str(p) + '_1.jpg', after)
			
			```
	- 运行效果（不同压缩比下）
		- ![](https://img-blog.csdnimg.cn/20190403134647184.png)
- 补充说明
	- 参考书为《Python3数据分析与机器学习实战》，对部分错误修改
	- 具体数据集和代码见我的Github，欢迎Star或者Fork
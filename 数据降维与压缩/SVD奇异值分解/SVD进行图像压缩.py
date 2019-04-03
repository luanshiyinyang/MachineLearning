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

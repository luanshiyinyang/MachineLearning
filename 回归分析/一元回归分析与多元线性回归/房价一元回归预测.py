# -*-coding:utf-8-*-
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np


# 输入文件，将房间大小和价格的数据转成scikitlearn中LinearRegression模型识别的数据
def get_data(file_name):
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    for single_square_meter, single_price_value in zip(data['square_meter'], data['price']):
        X_parameter.append([float(single_square_meter)])
        Y_parameter.append([float(single_price_value)])
    return X_parameter, Y_parameter


# 线性分析
def line_model_main(X_parameter, Y_parameter, predict_square_meter):
    # 模型对象
    regr = LinearRegression()
    # 训练模型
    regr.fit(X_parameter, Y_parameter)
    # 预测数据
    predict_outcome = regr.predict(predict_square_meter)
    predictions = {}
    # 截距值
    predictions['intercept'] = regr.intercept_
    # 斜率值
    predictions['coefficient'] = regr.coef_
    # 预测值
    predictions['predict_value'] = predict_outcome
    return predictions


# 显示图像
def show_linear_line(X_parameter, Y_parameter):
    # 构造模型对象
    regr = LinearRegression()
    # 训练模型
    regr.fit(X_parameter, Y_parameter)
    # 绘制已知数据的散点图
    plt.scatter(X_parameter, Y_parameter, color='blue')
    # 绘制预测直线
    plt.plot(X_parameter, regr.predict(X_parameter), color='red', linewidth=4)
    plt.title('Predict the house price')
    plt.xlabel('square meter')
    plt.ylabel('price')
    plt.show()


# 主函数
def main():
    # 读取数据
    X, Y = get_data('data/house_price.csv')
    # 获取预测值，这里我们预测700平英尺的房子的房价
    predict_square_meter = np.array([300, 400]).reshape(-1, 1)
    result = line_model_main(X, Y, predict_square_meter)
    for key, value in result.items():
        print('{0}:{1}'.format(key, value))
    # 绘图
    show_linear_line(X, Y)


if __name__ == '__main__':
    main()

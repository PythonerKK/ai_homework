"""
    :author: 李沅锟
    :url: http://github.com/PythonerKK
    :copyright: © 2019 KK <705555262@qq.com>
    :数据分析(线性回归算法)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import  linear_model


data = pd.read_csv("./data/guangzhou_ok.csv", encoding="utf-8")
# 删除第一列
data = data.iloc[:, 1:]


features = data['房间数量']
prices = data['price']

features = np.array(features).reshape([len(features), 1])
prices = np.array(prices)
print(features)
print(prices)
# mix_feature = min(features)
# max_feature = max(prices)
# X=np.arange(mix_feature, max_feature).reshape([-1, 1])

clf =linear_model.LinearRegression()
clf.fit(features, prices)#调用线性回归模块，建立回归方程，拟合数据
#查看回归方程系数
print('Cofficients:',clf.coef_)
#查看回归方程截距
print('intercept',clf.intercept_)


# #3.可视化处理
# #scatter函数用于绘制数据 点，这里表示用红色绘制数据点；
# plt.scatter(features, prices, color='red')
# #plot函数用来绘制直线，这 里表示用蓝色绘制回归线；
# #xlabel和ylabel用来指定横纵坐标的名称
# plt.plot(X,clf.predict(X),color='blue')
# plt.xlabel('Area')
# plt.ylabel('Price')
# plt.show()

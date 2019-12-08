"""
    :author: 李沅锟
    :url: http://github.com/PythonerKK
    :copyright: © 2019 KK <705555262@qq.com>
    :数据分析(线性回归算法)
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from sklearn import  linear_model

def transform_to_onehot(data: DataFrame, *features) -> DataFrame:
    """把中文特征转为数值型特征
    """
    _data = data.copy()
    for feature in features:
        list_unique = _data.loc[:, feature].unique()
        for j in range(len(list_unique)):
            _data.loc[:, feature] = _data.loc[:, feature].apply(
                lambda x: j if x == list_unique[j] else x)
    return _data

# 选取的属性
labels = ['房间数量', '建筑面积', '客厅数量', '厨房数量', '厕所数量', '装修情况', '楼层', '配备电梯']

# 数据预处理
data = pd.read_csv("./data/guangzhou_ok.csv", encoding="utf-8")
data = data.iloc[:, 1:] # 删除第一列
data = data[data['district'].isin(['天河'])] # 天河区房价分析
data = transform_to_onehot(data, "装修情况", "楼层", "district", '房屋朝向')

# 提取特征属性和价格属性
features = np.array(data[labels]) # 特征属性
prices = [price / 1000 for price in data['price']] # 价格标准化

# 测试集训练集划分
features_train, features_test, prices_train, prices_test = train_test_split(
    features, prices, test_size=0.15, random_state=0)

# 建立线性回归模型
clf = linear_model.LinearRegression()
clf.fit(features_train, prices_train) #调用线性回归模块，建立回归方程，拟合数据


#使用模型进行预测
y_predict = clf.predict(features_test)

#计算模型的预测值与真实值之间的均方误差MSE
print(f"均方误差MSE:{mean_squared_error(prices_test, y_predict)}")

#查看回归方程系数
#print(f'回归方程系数:{clf.coef_}')

#查看回归方程截距
print(f'截距:{clf.intercept_}')

# 各属性相关性字典
result_dict = {}
for label, coef in zip(labels, clf.coef_):
    result_dict[label] = coef

# 相关性字典按照相关性降序排列
import operator
result_dict = dict(sorted(result_dict.items(),
                          key=operator.itemgetter(1), reverse=True))

print(f'相关性字典：{result_dict}')

# 选取正相关性前3的属性
attrs = []
for d in sorted(result_dict)[:3]:
    attrs.append(d)
print(f'具有正相关性前3的属性为:{",".join(attrs)}')

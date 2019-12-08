"""
    :author: 李沅锟
    :url: http://github.com/PythonerKK
    :copyright: © 2019 KK <705555262@qq.com>
    :数据分析(KNN算法)
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn import preprocessing
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split


data = pd.read_csv("./data/guangzhou_ok.csv", encoding="utf-8")
# 删除第一列
data = data.iloc[:, 1:]

def transform_to_onehot(data, *features) -> DataFrame:
    """把中文特征转为数值型特征"""
    _data = data.copy()
    for feature in features:
        list_unique = _data.loc[:, feature].unique()
        for j in range(len(list_unique)):
            _data.loc[:, feature] = _data.loc[:, feature].apply(
                lambda x: j if x == list_unique[j] else x)
    return _data

# 将中文标签转换为数值
data = transform_to_onehot(data, "装修情况", "楼层", "district", '房屋朝向')

# 使用标准化归一数据
#min_max_scaler = preprocessing.MinMaxScaler()
standard_scaler = preprocessing.StandardScaler()


# 提取特征
features = data.drop(['price', 'title', '房屋朝向', '楼层总数', '梯户比例'], axis=1)
# 提取价格
prices = [price / 1000 for price in data['price']] # 价格标准化
# 转换为numpy的array

# 特征归一化
features = np.array(standard_scaler.fit_transform(features))
prices = np.array(prices)


# 测试集训练集划分
x_train, x_test, y_train, y_test = train_test_split(
    features, prices, test_size=0.2, random_state=0)


# 使用KNN训练模型
regressor=KNeighborsRegressor()
regressor.fit(x_train,y_train)

# 预测测试集的数据
y_pred=regressor.predict(x_test)

# 计算均方误差 越小效果越好
label_mse = mean_squared_error(y_test, y_pred)
rmse = label_mse ** (1/2)
print(f"标准误差={rmse}")
print (f"均方误差={label_mse}")


def performance_metric(y_true, y_predict):
    """计算r2分数"""
    score = r2_score(y_true, y_predict)
    return score

print(f'R2分数：{round(performance_metric(y_test, y_pred), 4)}')

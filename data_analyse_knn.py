"""
    :author: 李沅锟
    :url: http://github.com/PythonerKK
    :copyright: © 2019 KK <705555262@qq.com>
    :数据分析(KNN算法)
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn import neighbors
from sklearn.metrics import make_scorer, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split, KFold, GridSearchCV


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


data = transform_to_onehot(data, "装修情况", "楼层", "district")

# 提取特征
features = data.drop(['price', 'title', '房屋朝向', '楼层总数', '梯户比例'], axis=1)
# 提取价格
prices = data['房屋朝向']
# 转换为numpy的array
features = np.array(features)
prices = np.array(prices)

# 测试集训练集划分
features_train, features_test, prices_train, prices_test = train_test_split(
    features, prices, test_size=0.2, random_state=0)


def performance_metric(y_true, y_predict):
    """计算r2分数"""
    score = r2_score(y_true, y_predict)
    return score


def fit_model(X, y):
    """测试模型"""
    knn = neighbors.KNeighborsClassifier()
    knn.fit(X, y)
    return knn


if __name__ == '__main__':
    optimal = fit_model(features_train, prices_train)
    predicted_value = optimal.predict(features_test)
    r2 = performance_metric(prices_test, predicted_value)
    print(r2)

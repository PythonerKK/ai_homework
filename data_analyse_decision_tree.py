"""
    :author: 李沅锟
    :url: http://github.com/PythonerKK
    :copyright: © 2019 KK <705555262@qq.com>
    :数据分析(决策树算法)
"""
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.ensemble import RandomForestRegressor
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
prices = data['price']
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
    cross_validator = KFold(10, shuffle=True) # 通过交叉认证缓解数据集过拟合的现象
    regressor = DecisionTreeRegressor() # 建立决策树回归模型
    params = {'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    scoring_fnc = make_scorer(performance_metric)
    # 通过GridSearchCV找到最优深度参数（基于输入数据[X,y] 利于网格搜索找到最优的决策树模型）
    grid = GridSearchCV(estimator=regressor, param_grid=params,
                        scoring=scoring_fnc, cv=cross_validator)
    # 网格搜索
    grid = grid.fit(X, y)
    return grid.best_estimator_


optimal = fit_model(features_train, prices_train)
# 输出最优模型的参数 'max_depth'
print(f"最优参数max_depth是:{optimal.get_params()['max_depth']}")

predicted_value = optimal.predict(features_test)
r2 = performance_metric(prices_test, predicted_value)

# 每次交叉验证得到的数据集不同，因此每次运行的结果也不一定相同
print('最优模型在测试数据上 R^2 分数 {: .4f}' .format(r2))

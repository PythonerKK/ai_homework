"""
    :author: 李沅锟
    :url: http://github.com/PythonerKK
    :copyright: © 2019 KK <705555262@qq.com>
    :数据可视化
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame

from matplotlib.font_manager import _rebuild
_rebuild() # 加载字体库

data = pd.read_csv("./data/guangzhou_ok.csv", encoding="utf-8")
plt.rcParams['font.sans-serif'] = ['simhei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_price_mean(data: DataFrame) -> dict:
    '''返回按区的平均价格字典'''
    district_list = list(set(data['district']))
    district_price_dict = {}
    for district in district_list:
        result = data[data['district'].isin([district])]
        district_price_dict[district] = round(result['price'].mean(), 2)
    del district_list
    return district_price_dict

district_price_dict = get_price_mean(data)

def draw_price_distplot():
    """广州二手房价格分布图"""
    # f, [ax1,ax2] = plt.subplots(1, 2, figsize=(15, 5))
    sns.distplot(data['price'],hist=True,kde=True,rug=True) # 前两个默认就是True,rug是在最下方显示出频率情况，默认为False
    # bins=20 表示等分为20份的效果，同样有label等等参数
    sns.kdeplot(data['price'], shade=True, color='b') # shade表示线下颜色为阴影,color表示颜色是红色
    plt.title("广州二手房价格分布图")
    plt.savefig('./graph/广州二手房价格分布图.pdf', dpi=200)


def draw_district_price_mean_barplot():
    """广州二手房区域价格平均图"""
    sns.rugplot(data['price']) # 在下方画出频率情况
    sns.barplot(x=list(district_price_dict.keys()),
                y=list(district_price_dict.values()),
                )
    plt.title("广州二手房区域价格平均图")
    plt.savefig('./graph/广州二手房区域价格平均图.pdf', dpi=200)


def draw_area_displot():
    """广州二手房面积图"""
    sns.distplot(data['建筑面积'], bins=20, color='purple')
    plt.title("广州二手房面积图")
    plt.savefig('./graph/广州二手房面积图.pdf', dpi=200)


def draw_area_and_price_scatterplot():
    """面积和价格的散点图"""
    sns.jointplot(x=data['建筑面积'], y=data['price'])
    plt.title("广州二手房面积和价格关系图")
    plt.savefig('./graph/广州二手房面积和价格关系图.pdf', dpi=200)


if __name__ == '__main__':
    #draw_price_distplot()
    #draw_district_price_mean_barplot()
    #draw_area_displot()
    draw_area_and_price_scatterplot()

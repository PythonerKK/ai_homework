"""
    :author: 李沅锟
    :url: http://github.com/PythonerKK
    :copyright: © 2019 KK <705555262@qq.com.com>
    :数据可视化
"""
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("guangzhou_ok.csv", encoding="utf-8")
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


f, [ax1,ax2] = plt.subplots(1, 2, figsize=(15, 5))
sns.distplot(data['price'],hist=True,kde=True,rug=True, ax=ax1) # 前两个默认就是True,rug是在最下方显示出频率情况，默认为False
# bins=20 表示等分为20份的效果，同样有label等等参数
sns.kdeplot(data['price'], shade=True, color='b', ax=ax1) # shade表示线下颜色为阴影,color表示颜色是红色
sns.rugplot(data['price'], ax=ax1) # 在下方画出频率情况
sns.distplot(data['建筑面积'], bins=20, ax=ax2, color='r')
sns.kdeplot(data['建筑面积'], shade=False, ax=ax2)
#plt.title("广州二手房价格直方图")
#plt.savefig('./广州二手房价格直方图.pdf', dpi=200)


#sns.regplot(x='建筑面积', y='price', data=data, ax=ax2)
plt.show()


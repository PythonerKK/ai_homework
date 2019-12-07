"""
    :author: 李沅锟
    :url: http://github.com/PythonerKK
    :copyright: © 2019 KK <705555262@qq.com>
    :数据预处理
"""
import pandas as pd

data = pd.read_csv("./data/guangzhou_all.csv", encoding="utf-8")
house_type = data['房屋户型']
print(data)
room_nums = [r.split('室')[0] for r in house_type]
living_room_nums = [lr.split('厅')[0].split('室')[1] for lr in house_type]
kitchen_nums = [k.split('厨')[0].split('厅')[1] for k in house_type]
wc_nums = [wc.split('厨')[1].split('卫')[0] for wc in house_type]
floor = [f.split('楼')[0] for f in data['所在楼层']]
stairs = [s.split('共')[1].split('层')[0] for s in data['所在楼层']]
areas = [a.split('㎡')[0] for a in data['建筑面积']]
elevator = []
for e in data['配备电梯']:
    if e == '有':
        elevator.append(1)
    else:
        elevator.append(0)

data['房间数量'] = room_nums
data['客厅数量'] = living_room_nums
data['厨房数量'] = kitchen_nums
data['厕所数量'] = wc_nums
data['楼层'] = floor
data['楼层总数'] = stairs
data['建筑面积'] = areas
del data['房屋户型']
del data['所在楼层']
del data['户型结构']
del data['套内面积']
del data['建筑类型']
del data['建筑结构']
del data['产权年限']
del data['url']
data['配备电梯'] = elevator
data = data.sort_values('price', ascending=True)

#data = data.iloc[:, 1:]
print(data)
data.to_csv("./data/guangzhou_ok.csv", encoding="utf-8")

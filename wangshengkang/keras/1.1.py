# ----------------------开发者信息-----------------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/5/20 15:45
# @Author: wangshengkang
# @Version: 1.0
# @FileName: 1.0.py
# @Software: PyCharm
# ----------------------开发者信息-----------------------------------------

# ----------------------代码布局-------------------------------------
# 1.引入keras，matplotlib，numpy，sklearn，pandas包
# 2.导入数据
# 3.数据归一化
# 4.模型建立
# 5.损失函数可视化
# 6.预测结果
# ---------------------------------------------------------------------

# ---------------------------1引入相关包--------------------------------
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model

# -----------------------------1引入相关包-----------------------------

# ------------------------------2导入数据-----------------------------
data = np.load('boston_housing.npz')  # 读取数据
trainx = data['x'][0:404]  # 前404个数据为训练集
trainy = data['y'][0:404]
validx = data['x'][404:]  # 后面的数据为测试集
validy = data['y'][404:]
data.close()

# 将数据转化为DataFrame形式
trainx_pd = pd.DataFrame(trainx)
trainy_pd = pd.DataFrame(trainy)
validx_pd = pd.DataFrame(validx)
validy_pd = pd.DataFrame(validy)

# 查看训练集前五条数据
print(trainx_pd.head(5))
print(trainy_pd.head(5))
# -------------------------------2导入数据--------------------------------

# -------------------------------3数据归一化------------------------------
# 正则化
minmaxscale = MinMaxScaler()
minmaxscale.fit(trainx_pd)  # fit用来获取最大值最小值
trainxg = minmaxscale.transform(trainx_pd)  # transform根据min和max来scale
minmaxscale.fit(trainy_pd)
trainyg = minmaxscale.transform(trainy_pd)
minmaxscale.fit(validx_pd)
validxg = minmaxscale.transform(validx_pd)
minmaxscale.fit(validy_pd)
validyg = minmaxscale.transform(validy_pd)
# ----------------------------3数据归一化-----------------------------

# -----------------------------4模型建立-------------------------------

model = Sequential()  # 创建序列化模型
# Dense层，第一层，注意input_shape里面的逗号
model.add(Dense(units=10, activation='relu', input_shape=(trainx_pd.shape[1],)))
model.add(Dropout(0.2))
model.add(Dense(units=15, activation='relu'))
model.add(Dense(units=1, activation='linear'))

print(model.summary())  # 打印模型

model.compile(optimizer='adam', loss='mse')  # 设置模型优化器，损失函数

# 训练模型，返回值为一个history对象，里面记录了损失值
result = model.fit(trainxg, trainyg, batch_size=200, epochs=200, verbose=2,
                   validation_data=(validxg, validyg))
# ------------------------------4模型建立----------------------------------

# ------------------------------5损失函数可视化----------------------------
plt.plot(result.history['loss'])  # 从history对象中获取训练集损失
plt.plot(result.history['val_loss'])  # 从history对象中获取验证集损失
plt.title('The Loss')  # 题目
plt.xlabel('epoch')  # 横坐标
plt.ylabel('Loss')  # 纵坐标
plt.legend(['train', 'validation'], loc='upper left')  # 曲线名字
plt.show()  # 画图
# ------------------------------5损失函数可视化------------------------------

# ------------------------------6保存模型预测结果----------------------------
model.save('11result.h5')  # 保存模型
plot_model(model, to_file='modelimage.jpg', show_shapes=True)  # 将模型的结构画出来
model = load_model('11result.h5')  # 加载模型
validgpre = model.predict(validxg)  # 用归一化后的数据进行预测
minmaxscale.fit(validy_pd)  # 获取原始数据min和max
pre = minmaxscale.inverse_transform(validgpre)  # 根据min和max进行scale
pre_pd = pd.DataFrame(pre)  # 将预测结果转换为DataFrame形式
print(pre_pd.head(5))  # 查看预测结果前五个
# ------------------------------6保存模型预测结果--------------------------

# -*- coding: utf-8 -*-
# @Time: 2020/5/29 14:32
# @Author: wangshengkang
# @Version: 1.0
# @Filename: 22imagecnntorch.py
# @Software: PyCharm
# -------------------------------------代码布局：---------------------------------------
# 1引入gzip,numpy,keras,os等包
# 2导入数据，处理数据
# 3创建模型
# 4训练模型
# 5保存模型
# 6画图
# ------------------------------------1引入相关包--------------------------------------
import gzip
import numpy as np
import os
import torch
import torch.nn as nn


# ------------------------------------1引入相关包----------------------------------
# -----------------------------------2导入数据，数据处理------------------------------------------
def load_data():
    paths = [
        'train-labels-idx1-ubyte.gz',
        'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz',
        't10k-images-idx3-ubyte.gz'
    ]
    with gzip.open(paths[0], 'rb') as lbpath:
        # frombuffer将data以流的形式读入转化成ndarray对象
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)
    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)
    return (x_train, y_train), (x_test, y_test)


(x_train, y_train), (x_test, y_test) = load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255  # 归一化
x_test /= 255  # 归一化
x_train = torch.LongTensor(x_train)
y_train = torch.LongTensor(y_train)
x_test = torch.LongTensor(x_test)
y_test = torch.LongTensor(y_test)

epochs = 5
data_augmentation = True
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_cnn')
model_name = 'keras_fashion_trained_model.h5'
# -----------------------------------2导入数据，数据处理------------------------------------------
# -----------------------------------3创建模型----------------------------------------------
class image(nn.Module):
    def __init__(self):
        super(image, self).__init__()
        self.conv2d1 = nn.Conv2d(1, 32, (3, 3), padding='1')  # 32*28*28
        self.relu1 = nn.ReLU()
        self.conv2d2 = nn.Conv2d(32, 32, (3, 3))  # 32*26*26
        self.relu2 = nn.ReLU()
        self.maxpooling2d1 = nn.MaxPool2d((2, 2))  # 32*13*13
        self.dropout1 = nn.Dropout(0.25)

        self.conv2d3 = nn.Conv2d(64, (3, 3), padding='1')  # 64*13*13
        self.relu3 = nn.ReLU()
        self.con2d4 = nn.Conv2d(64, (3, 3), padding='1')  # 64*13*13
        self.relu4 = nn.ReLU()
        self.maxpooling2d2 = nn.MaxPool2d((2, 2))  # 64*6*6
        self.dropout2 = nn.Dropout()

        self.flatten = nn.Flatten()  # 2304
        self.fc1 = nn.Linear(2304, 512)  # 512
        self.relu5 = nn.ReLU()
        self.dropout3 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)  # 10
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv2d1(x)
        x = self.relu1(x)
        x = self.conv2d2(x)
        x = self.relu2(x)
        x = self.maxpooling2d1(x)
        x = self.dropout1(x)

        x = self.conv2d3(x)
        x = self.relu3(x)
        x = self.conv2d4(x)
        x = self.relu4(x)
        x = self.maxpooling2d2(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.dropout3(x)
        x = self.fc2(x)
        x = self.softmax(x)


model = image()
optimizer = torch.optim.Adam(lr=0.0001)
loss = nn.CrossEntropyLoss()
for epoch in range(epochs):
    pre = model(x_train)
    batch_loss = loss(pre, y_train)
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()
    print('epoch %d , loss %10f' % (epoch, batch_loss))

# -----------------------------------3创建模型----------------------------------------------

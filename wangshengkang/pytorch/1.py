#------------------------作者信息-----------------------------------------
# -*- coding: utf-8 -*-
# @Time: 2020/5/23 20:04
# @Author: wangshengkang
# @Version: 1.0
# @Filename: 1.py
# @Software: PyCharm
#--------------------------作者信息--------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.autograd import Variable

data=np.load('../boston_housing.npz')
train_x=data['x'][:404]
train_y=data['y'][:404]
valid_x=data['x'][404:]
valid_y=data['y'][404:]

train_x_pd=pd.DataFrame(train_x)
train_y_pd=pd.DataFrame(train_y)
valid_x_pd=pd.DataFrame(valid_x)
valid_y_pd=pd.DataFrame(valid_y)

print(train_x_pd.head(5))
print(train_y_pd.head(5))

min_max_scale=MinMaxScaler()
min_max_scale.fit(train_x_pd)
train_x_sc=min_max_scale.transform(train_x_pd)
#my_array1 = np.array(train_x_sc)
#my_tensor1 = torch.tensor(my_array1).float()
#x=Variable(my_tensor1,requires_grad=True)

#x=torch.from_numpy(train_x_sc).float()
x = torch.autograd.Variable(torch.from_numpy(train_x_sc))
x=x.float()

min_max_scale.fit(train_y_pd)
train_y_sc=min_max_scale.transform(train_y_pd)
#my_array2 = np.array(train_y_sc)
#my_tensor2 = torch.tensor(my_array2).float()
#y=Variable(my_tensor2,requires_grad=True)

#y=torch.from_numpy(train_y_sc).float()
y = torch.autograd.Variable(torch.from_numpy(train_y_sc))
y=y.float()



min_max_scale.fit(valid_x_pd)
valid_x_sc=min_max_scale.transform(valid_x_pd)
min_max_scale.fit(valid_y_pd)
valid_y_sc=min_max_scale.transform(valid_y_pd)

class house(nn.Module):
    def __init__(self):
        super(house,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(13,10),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(10,15),
            nn.ReLU(),

            nn.Linear(15,1),
        )

    def forward(self,x):
        out = self.fc(x)
        return out

model=house()
loss=nn.MSELoss(reduction='sum')
#loss=nn.MSELoss(reduce=True, size_average=True)
optimizer= torch.optim.Adam(model.parameters(),lr=0.001)
epochs=200

for epoch in range(epochs):
    train_loss=0.0
    model.train()
    train_pre=model(x)
    batch_loss=loss(y,train_pre)
    optimizer.zero_grad()
    batch_loss.backward()
    optimizer.step()
    print('epoch %3d , loss %3d ' % (epoch,batch_loss))




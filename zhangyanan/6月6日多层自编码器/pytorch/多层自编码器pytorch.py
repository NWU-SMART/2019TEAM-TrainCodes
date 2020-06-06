# ----------------开发者信息--------------------------------#
# 开发者：张亚楠
# 开发日期：2020年6月6日
# 修改日期：
# 修改人：
# 修改内容：
'''使用gpu训练'''

#  -------------------------- 导入需要包 -------------------------------
import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt



#  --------------------- 读取手写数据集并对图像预处理 ---------------------
path = 'mnist.npz'
f = np.load(path)
####  以npz结尾的数据集是压缩文件，里面还有其他的文件
####  使用：f.files 命令进行查看,输出结果为 ['x_test', 'x_train', 'y_train', 'y_test']
# 60000个训练，10000个测试
# 训练数据
X_train=f['x_train']
# 测试数据
X_test=f['x_test']
f.close()


# 观察下X_train和X_test维度
print(X_train.shape)  # 输出X_train维度  (60000, 28, 28)
print(X_test.shape)   # 输出X_test维度   (10000, 28, 28)

# 数据预处理
#  归一化
X_train = X_train.astype("float32")/255.
X_test = X_test.astype("float32")/255.

print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

##### --------- 输出语句结果 --------
#    X_train shape: (60000, 28, 28)
#    60000 train samples
#    10000 test samples
##### --------- 输出语句结果 --------

# 数据准备

# np.prod是将28X28矩阵转化成1X784，方便BP神经网络输入层784个神经元读取
# len(X_train) --> 6000, np.prod(X_train.shape[1:])) 784 (28*28)
# X_train 60000*784, X_test10000*784
X_train = X_train.reshape((len(X_train), np.prod(X_train.shape[1:])))
X_test = X_test.reshape((len(X_test), np.prod(X_test.shape[1:])))

# 把x_train,x_test变为tensor格式
X_train = torch.Tensor(X_train)
X_test = torch.Tensor(X_test)



#  --------------------- 构建单层自编码器模型 ---------------------


# 定义神经网络层数
class EncoderModel(nn.Module):
    def __init__(self):
        super(EncoderModel, self).__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(784, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 784),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        output = self.dense(x)
        return output

model = EncoderModel()

# 调用GPU加速训练，也就是在模型，x_train后面加上cuda()
model = model.cuda()
X_train = X_train.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torch.nn.MSELoss()
Epoch = 5

# 使用dataloader载入数据，小批量进行迭代
import torch.utils.data as Data
torch_dataset = Data.TensorDataset(X_train, X_train)
loader = Data.DataLoader(dataset=torch_dataset, batch_size=1024, shuffle=True, num_workers=0)


loss_list = [] # 建立一个loss的列表，以保存每一次loss的数值
for t in range(5):
    running_loss = 0
    for step, (X_train, X_train) in enumerate(loader):
        train_prediction = model(X_train)
        loss = loss_func(train_prediction, X_train)  # 计算损失
        loss_list.append(loss) # 使用append()方法把每一次的loss添加到loss_list中
        optimizer.zero_grad()  # 由于pytorch的动态计算图，所以在进行梯度下降更新参数的时候，梯度并不会自动清零。需要在每个batch候清零梯度
        loss.backward()  # 反向传播，计算参数
        optimizer.step()  # 更新参数
        running_loss += loss.item()
    else:
        print(f"第{t}代训练损失为：{running_loss/len(loader)}")  # 打印平均损失

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.plot(loss_list, 'c-')
plt.show()

















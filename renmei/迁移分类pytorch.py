#=================2020.06.09======================
# ----------------------   代码布局： ----------------------
# 1、导入 包
# 导入数据及与图像预处理
# 3、迁移学习建模
# 4、训练
#
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
from tensorflow.python.keras.utils import get_file
import gzip
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt
import time
import copy
from PIL import Image
#  --------------------- 2、读取数据及与图像预处理 ---------------------

path = 'D:\\keras_datasets\\'


# 数据集与代码放在一起即可
def load_data():
    paths = [
        path + 'train-labels-idx1-ubyte.gz', path + 'train-images-idx3-ubyte.gz',
        path + 't10k-labels-idx1-ubyte.gz', path + 't10k-images-idx3-ubyte.gz'
    ]

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_train), 28, 28, 1)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(), np.uint8, offset=8)

    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(
            imgpath.read(), np.uint8, offset=16).reshape(len(y_test), 28, 28, 1)

    return (x_train, y_train), (x_test, y_test)


# read dataset
(x_train, y_train), (x_test, y_test) = load_data()
batch_size = 32
num_classes = 10
epochs = 5
data_augmentation = True  # 图像增强
num_predictions = 20
save_dir = os.path.join(os.getcwd(), 'saved_models_transfer_learning')
model_name = 'pytorch_fashion_transfer_learning_trained_model.h5'
X_train = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_train]
X_test = [cv2.cvtColor(cv2.resize(i, (48, 48)), cv2.COLOR_GRAY2RGB) for i in x_test]

x_train = np.asarray(X_train)
x_test = np.asarray(X_test)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

x_train /= 255  # 归一化
x_test /= 255  # 归一化
base_model=models.vgg16(include_top=False, weights='imagenet', input_shape=x_train.shape[1:])
#-------------------------------------建立模型-----------------
class Net(nn.Module):
    def __init__(self,models):
        self.models=models
        super(Net, self).__init__()
        self.inputs=inputs
        self.output=outputs
        self.linear=nn.Sequential(
            nn.Linear(7*7*512,256),
            nn.Relu(),
            nn.Dropout(0.5)
        )
        self.linear2=nn.Sequential(
            nn.Linear(512,num_classes),
            nn.Softmax()
        )
    y=base_model.output_shape[1:]
    def forward(self,y):
        x=self.linear(base_model.output_shape[1:])
        out=self.linear2(x)
        return out
model=Net()
#------------------------------------训练------
for epoch in range(n_epochs):
    since = time.time()
    print("Epoch{}/{}".format(epoch, n_epochs))
    print("-" * 10)
    for param in ["train", "val"]:
        if param == "train":
            model.train = True
        else:
            model.train = False

        running_loss = 0.0
        running_correct = 0
        batch = 0
        for data in data_loader_image[param]:
            batch += 1
            X, y = data
            if use_gpu:
                X, y = Variable(X.cuda()), Variable(y.cuda())
            else:
                X, y = Variable(X), Variable(y)

            optimizer.zero_grad()
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)

            loss = cost(y_pred, y)
            if param == "train":
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            # running_loss += loss.data[0]
            running_correct += torch.sum(pred == y.data)
            if batch % 5 == 0 and param == "train":
                print("Batch {}, Train Loss:{:.4f}, Train ACC:{:.4f}".format(
                    batch, running_loss / (4 * batch), 100 * running_correct / (4 * batch)))

        epoch_loss = running_loss / len(data_image[param])
        epoch_correct = 100 * running_correct / len(data_image[param])

        print("{} Loss:{:.4f}, Correct:{:.4f}".format(param, epoch_loss, epoch_correct))
    now_time = time.time() - since
    print("Training time is:{:.0f}m {:.0f}s".format(now_time // 60, now_time % 60))

torch.save(model, 'model.pth')


loss=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(net.parameters(),lr=lr)







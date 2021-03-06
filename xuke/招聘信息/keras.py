#--------------         开发者信息--------------------------
#开发者：徐珂
#开发日期：2020.5.29
#software：pycharm
#项目名称：招聘信息（keras）


# ----------------------   代码布局： ----------------------
# 1、导入 Keras, matplotlib, numpy, sklearn 和 panda的包
# 2、招聘数据数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
# ----------------------   代码布局： ----------------------

#  -------------------------- 1、导入需要包 -------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers  # 正则化
import matplotlib.pyplot as plt
import numpy as np
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from keras.layers import BatchNormalization

#  -------------------------- 1、导入需要包 -------------------------------


#  -------------------------- 2、招聘数据数据导入 -------------------------------
# 文件放置在目录  D:\keras\招聘信息\job_detail_dataset.csv
job_detail_pd = pd.read_csv('job_detail_dataset.csv', encoding='UTF-8')
print(job_detail_pd.head(5))                          # 打印前五行 #
label = list(job_detail_pd['PositionType'].unique())  # 标签 #
print(label)

# 为工作描述设置标签的id #
def label_dataset(row):
    num_label = label.index(row)        # 返回label列表对应值的索引 #
    return num_label
job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)
job_detail_pd = job_detail_pd.dropna()  # 删除空行 #
job_detail_pd.head(5)
#  -------------------------- 2、招聘数据数据导入 -------------------------------


#  -------------------------- 3、分词和提取关键词 -------------------------------
# 中文分词
def chinese_word_cut(row):
    return " ".join(jieba.cut(row))
job_detail_pd['Job_Description_jieba_cut'] = job_detail_pd.Job_Description.apply(chinese_word_cut)
job_detail_pd.head(5)

# 提取关键词
def key_word_extract(texts):
    return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
job_detail_pd['Job_Description_key_word'] = job_detail_pd.Job_Description.apply(key_word_extract)
#  -------------------------- 3、分词和提取关键词 -------------------------------


#  -------------------------- 4、建立字典，并使用 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现次数排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50  词嵌入前的预处理
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)  # 长度都填充到50
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()  # 把数组转化为列表
#  -------------------------- 4、建立字典，并使用 -------------------------------



#  ------------------------------- 5、模型建立   ----------------------------------

#  -------------------------- 5.1、Sequential()    -------------------------------
model = Sequential()
model.add(Embedding(output_dim=32,     # 词向量的维度
                    input_dim=2000,    # Size of the vocabulary 字典大小
                    input_length=50    # 每个数字列表的长度
                    )
          )

model.add(Dropout(0.2))                # 丢掉20%神经元
model.add(Flatten())                   # 平展
model.add(Dense(units=256,activation="relu"))    #输入维度256，relu激活
model.add(Dropout(0.25))
model.add(Dense(units=10,activation="softmax"))
print(model.summary())  # 打印模型


#  -------------------------- 5.2、API    -------------------------------
inputs=Input(shape=())
x = Embedding(2000,32)(inputs)    # 嵌入第一层 #
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense1(256, activation='relu')(x)
x = Dropout(0.25)(x)
outputs = Dense2(10, activation='softmax')(x)
model = Model(inputs=inputs, outputs=outputs)

# -------------------------- 5.3、model类继承--------------------------#
class Model(keras.Model):
    def __init__(self, use_dp=True):
        super(HousePredict, self).__init__(name='mlp')
        self.Embedding = keras.layers.Embedding(2000,32)
        self.dropout = keras.layers.Dropout(0.2)
        self.Flatten = keras.layers.Flatten()
        self.dense1 = keras.layers.Dense(256,activation='relu')
        self.dropout = keras.layers.Dropout(0.25)
        self.dense2 = keras.layers.Dense(10, activation='softmax')
    def call(self,inputs):
         x = self.Embedding(inputs)
         x = self.dropout(x)
         x = self.Flatten(x)
         x = self.dense1(x)
         x = self.dropout(x)
         x = self.dense2(x)
         return x
    model = Model()
    inputs = Input(shape=())


#  ------------------------------- 5、模型建立   ----------------------------------

#  -------------------------- 6、训练模型 -------------------------------
batch_size = 256
epochs = 5
model = Sequential()
model.add(Embedding(output_dim=32,  # 词向量的维度
                    input_dim=2000,  # Size of the vocabulary 字典大小
                    input_length=50  # 每个数字列表的长度
                    )
          )
model.add(Dropout(0.2))
model.add(Flatten())  # 平展
model.add(Dense(units=256,activation="relu"))
model.add(Dropout(0.25))
model.add(Dense(units=10,activation="softmax"))
print(model.summary())  # 打印模型
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam",metrics=["acc"])
history = model.fit(x_train,y_train,batch_size=batch_size,epochs=epochs,verbose=2,validation_split=0.2)  # 训练集的20%用作验证集
#  -------------------------- 6、训练模型 -------------------------------


#  -------------------------- 7、保存模型，显示运行结果 -------------------------------
from keras.utils import plot_model

# 保存模型
model.save('model_MLP_text.h5')  # 生成模型文件 'my_model.h5'
# 模型可视化
plot_model(model, to_file='model_MLP_text.png', show_shapes=True)

from keras.models import load_model

# 加载模型
# model = load_model('model_MLP_text.h5')
print(x_train[0])
y_new = model.predict(x_train[0].reshape(1, 50))
print(list(y_new[0]).index(max(y_new[0])))
print(y_train[0])

import matplotlib.pyplot as plt

# 绘制训练 & 验证的准确率值
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_acc.png')
plt.show()

# 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_loss.png')
plt.show()
#  -------------------------- 7、保存模型，显示运行结果 -------------------------------
# /------------------ 开发者信息 --------------------/
# /** 开发者：于林生
#
# 开发日期：2020.5.20
#
# 版本号：Versoin 1.0
#
# 修改日期：
#
# 修改人：
#
# 修改内容： / /------------------ 开发者信息 --------------------*/

# 导入需要的包
import pandas as pd #数据处理包
import jieba #中文词库包
import jieba.analyse as analyse
import numpy as np
import matplotlib.pyplot as plt #画图包
from sklearn.model_selection import train_test_split #数据划分包
from sklearn.preprocessing import LabelEncoder  #数据有类别编码
# 导入需要的keras包
from keras.preprocessing.text import Tokenizer  #将文本处理成索引类型的数据
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
from keras.utils import multi_gpu_model
from keras.models import load_model
from keras import regularizers
from keras.utils import plot_model
from keras.utils.np_utils import to_categorical
from keras.layers import BatchNormalization


# 读取数据
# 数据路径
path = 'G:/python/code/多层感知器/pytorch改编代码/job_detail_dataset.csv'
# 读取数据
data = pd.read_csv(path,encoding='utf-8')
# 维度（50000 x 2 ）
print(data)#PositionType Job_Description


# 数据处理
label = list(data['PositionType'].unique())#去重读取工作类型 10类
print(label)
print(label.index('项目管理'))#找到项目管理的索引（list的属性）
# 为工作描述设置标签的id
def label_dataset(row):
     num_label = label.index(row)  # 返回label列表对应值的索引
     return num_label

# 给不同的工作类型打上分类标签
data['label'] = data['PositionType'].apply(label_dataset)
print(data)
data = data.dropna()
print(data)#44831*3

# 提取描述中的中文分词并写入
# 采用的精确模式  他来到上海交通大学  ->   他/ 来到/ 上海交通大学
# (若参数cut_all=True)  ->  他/ 来到/ 上海/ 上海交通大学/ 交通/ 大学
def chinese_word(row):
    return " ".join(jieba.cut(row))
data['chinese_cut'] = data.Job_Description.apply(chinese_word)
data.head(5)


# 提取关键词
# 提取关键词
# analyse.extract_tags(texts,topK,withWeight,allowPOS)
'''
第一个参数：待提取关键词的文本
第二个参数：返回关键词的数量，重要性从高到低排序
第三个参数：是否同时返回每个关键词的权重
第四个参数：词性过滤，为空表示不过滤，若提供则仅返回符合词性要求的关键词
'''
def key_word_extract(texts):
  return " ".join(analyse.extract_tags(texts, topK=50, withWeight=False, allowPOS=()))
data['keyword']=data.Job_Description.apply(key_word_extract)
data.head(5)

# 建立字典
token = Tokenizer(num_words=1000)
# 按照单词出现的顺序建立
token.fit_on_texts(data['keyword'])

description = token.texts_to_sequences(data['keyword'])
job_description = sequence.pad_sequences(description,maxlen=50)
# 选取训练集
x_train = job_description
y_train = data['label'].tolist()
print(x_train)
print(y_train)


# cnn模型
model =  Sequential()
model.add(Embedding(1000,32,input_length=50))
model.add(Conv1D(256,3,padding='same',activation='relu'))
model.add(MaxPool1D(3,3,padding='same'))
model.add(Conv1D(32,3,padding='same',activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(BatchNormalization())
model.add(Dense(256,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(units=10,activation='softmax'))

# 模型训练
model.compile(loss='sparse_categorical_crossentropy',
             optimizer='adam',
             metrics = ['accuracy'])
result_cnn = model.fit(x_train,y_train,batch_size=256,epochs=5,verbose=2,validation_split=0.2)


# 画图
import matplotlib.pyplot as plt
plt.plot(result_cnn.history['loss'])
plt.plot(result_cnn.history['val_loss'])

# plt.plot（result.result['val_accuracy']）
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
plt.plot(result_cnn.history['accuracy'])
plt.plot(result_cnn.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()
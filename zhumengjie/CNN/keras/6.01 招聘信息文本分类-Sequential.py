# 开发者：朱梦婕
# 开发日期：2020年5月30日
# 开发框架：keras
#----------------------------------------------------------#

# ----------------------   代码布局： ---------------------- #
# 1、导入 Keras, matplotlib, numpy, sklearn panda jieba的包
# 2、招聘数据数据导入
# 3、分词和提取关键词
# 4、建立字典，并使用
# 5、训练模型
# 6、保存模型，显示运行结果
#--------------------------------------------------------------#

#  -------------------------- 1、导入需要包 -------------------------------
import pandas as pd
import jieba
import jieba.analyse as analyse
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, MaxPool1D, Conv1D
from keras.layers.embeddings import Embedding
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.layers import BatchNormalization

#  -------------------------- 导入需要包 -------------------------------

#  -------------------------- 2、招聘数据数据导入 -------------------------------
# 文件放置在目录  D:\keras_datasets\job_detail_dataset.csv
job_detail_pd = pd.read_csv('E:\study\kedata\job_detail_dataset.csv', encoding='UTF-8') # 数据读取
print(job_detail_pd.head(5))  # 显示前5个数据
label = list(job_detail_pd['PositionType'].unique())  # 标签
print(label) # 输出标签


# 为工作描述设置标签的id
def label_dataset(row):
    num_label = label.index(row)  # 返回label列表对应值的索引
    return num_label


job_detail_pd['label'] = job_detail_pd['PositionType'].apply(label_dataset)
job_detail_pd = job_detail_pd.dropna()  # 删除空行
job_detail_pd.head(5)


#  -------------------------- 招聘数据数据导入 -------------------------------

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
#  --------------------------分词和提取关键词 -------------------------------

#  -------------------------- 4、建立字典，并使用 -------------------------------
# 建立2000个词的字典
token = Tokenizer(num_words=2000)
token.fit_on_texts(job_detail_pd['Job_Description_key_word'])  # 按单词出现次数排序，排序前2000的单词会列入词典中

# 使用token字典将“文字”转化为“数字列表”
Job_Description_Seq = token.texts_to_sequences(job_detail_pd['Job_Description_key_word'])

# 截长补短让所有“数字列表”长度都是50
Job_Description_Seq_Padding = sequence.pad_sequences(Job_Description_Seq, maxlen=50)
x_train = Job_Description_Seq_Padding
y_train = job_detail_pd['label'].tolist()
#  -------------------------- 建立字典，并使用 -------------------------------

#  -------------------------- 5、训练模型 -------------------------------
model = Sequential()
model.add(Embedding(output_dim=32,  # 词向量的维度
                    input_dim=2000,  # Size of the vocabulary 字典大小
                    input_length=50  # 每个数字列表的长度
                    )
          )

model.add(Conv1D(256,  # 输出大小
                 3,  # 卷积核大小
                 padding='same',
                 activation='relu'))
model.add(MaxPool1D(3, 3, padding='same'))
model.add(Conv1D(32, 3, padding='same', activation='relu'))
model.add(Flatten())
model.add(Dropout(0.3))
model.add(BatchNormalization())  # (批)规范化层
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=10,
                activation="softmax"))

batch_size = 256
epochs = 5

# 单GPU版本
model.summary()  # 可视化模型
model.compile(loss="sparse_categorical_crossentropy",  # 多分类
              optimizer="adam",
              metrics=["accuracy"])

history = model.fit(
    x_train,
    y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.2
    # 训练集的20%用作验证集
)
#  -------------------------- 5、训练模型 -------------------------------


#  -------------------------- 6、保存模型，显示运行结果 -------------------------------
# 保存模型
model.save('model_CNN_text.h5')  # 生成模型文件 'my_model.h5'
# 模型可视化
plot_model(model, to_file='model_CNN_text.png', show_shapes=True)


# 加载模型
# model = load_model('model_CNN_text.h5')
print(x_train[0])
y_new = model.predict(x_train[0].reshape(1, 50))
print(list(y_new[0]).index(max(y_new[0])))
print(y_train[0])


# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper left')
plt.savefig('Valid_accuracy.png')
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

#  -------------------------- 6、保存模型，显示运行结果 -------------------------------
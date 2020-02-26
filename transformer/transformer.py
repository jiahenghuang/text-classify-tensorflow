#coding:utf-8
import os
import csv
import time
import datetime
import random
import json
from collections import Counter
from math import sqrt
import gensim
import pandas as pd
import numpy as np
import tensorflow as tf

from tensorflow.keras import backend as K
from tensorflow.keras.layers import GlobalAveragePooling1D,TimeDistributed,BatchNormalization,Layer,Input,Conv2D,MaxPool2D,concatenate,Flatten,Dense,Dropout,Embedding,Reshape,LSTM
from tensorflow.keras import Sequential,optimizers,losses
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import regularizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import initializers
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Flatten
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from bs4 import BeautifulSoup
import logging
import gensim
from gensim.models import word2vec

import multiprocessing
import yaml
import jieba

class Config(object):
    #数据集路径
    dataSource = "../data/goods_zh.txt"
    stopWordSource = "../data/stopword.txt"
    
    #分词后保留大于等于最低词频的词
    miniFreq=1
    
    #统一输入文本序列的定长，取了所有序列长度的均值。超出将被截断，不足则补0
    sequenceLength = 200  
    batchSize=64
    epochs=10
    
    numClasses = 2
    #训练集的比例
    rate = 0.8  
    
    #生成嵌入词向量的维度
    embeddingSize = 150
    
    #卷积核数
    numFilters = 30
    
    #卷积核大小
    filterSizes = [2,3,4,5]
    dropoutKeepProb = 0.5
    
    #L2正则系数
    l2RegLambda = 0.1
        
# 实例化配置参数对象
config = Config()

# 设置输出日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

file = open("../data/goods_zh.txt") 
sentences=[]
for line in file:
    temp=line.replace('\n','').split(',,')
    sentences.append(jieba.lcut(temp[0]))
file.close()

model = word2vec.Word2Vec(sentences,size=110,
                     min_count=1,
                     window=10,
                     workers=multiprocessing.cpu_count(),sg=1,
                     iter=20)
model.save('../data/word2VecModel')
model = gensim.models.Word2Vec.load('../data/word2VecModel')
# 数据预处理的类，生成训练集和测试集
class Dataset(object):
    def __init__(self, config):
        self.dataSource = config.dataSource
        self.stopWordSource = config.stopWordSource  
        
        # 每条输入的序列处理为定长
        self.sequenceLength = config.sequenceLength  
        
        self.embeddingSize = config.embeddingSize
        self.batchSize = config.batchSize
        self.rate = config.rate
        self.miniFreq=config.miniFreq
        
        self.stopWordDict = {}
        
        self.trainReviews = []
        self.trainLabels = []
        
        self.evalReviews = []
        self.evalLabels = []
        
        self.wordEmbedding =None
        self.n_symbols=0
        
        self.wordToIndex = {}
        self.indexToWord = {}
        
    def readData(self, filePath):
        file = open(filePath) 
        text=[]
        label=[]
        for line in file:
            temp=line.replace('\n','').split(',,')
            text.append(temp[0])
            label.append(temp[1])
        file.close()
        
        print('data:',len(text),len(label))
        texts = [jieba.lcut(document.replace('\n', '')) for document in text]

        return texts, label

    def readStopWord(self, stopWordPath):
        """
        读取停用词
        """
        with open(stopWordPath, "r") as f:
            stopWords = f.read()
            stopWordList = stopWords.splitlines()
            # 将停用词用列表的形式生成，之后查找停用词时会比较快
            self.stopWordDict = dict(zip(stopWordList, list(range(len(stopWordList)))))
    
    def getWordEmbedding(self, words):
        """
        按照我们的数据集中的单词取出预训练好的word2vec中的词向量
        """
        
        #中文
        model = gensim.models.Word2Vec.load('../data/word2VecModel')
        
        vocab = []
        wordEmbedding = []
        
        # 添加 "pad" 和 "UNK", 
        vocab.append("pad")
        wordEmbedding.append(np.zeros(self.embeddingSize))
        
        vocab.append("UNK")
        wordEmbedding.append(np.random.randn(self.embeddingSize))
        
        for word in words:
            try:
                #中文
                vector =model[word]
                
                vocab.append(word)
                wordEmbedding.append(vector)
            except:
                
                print(word + " : 不存在于词向量中")
                
        return vocab, np.array(wordEmbedding)
    
    def genVocabulary(self, reviews):
        """
        生成词向量和词汇-索引映射字典，可以用全数据集
        """
        allWords = [word for review in reviews for word in review]
        # 去掉停用词
        subWords = [word for word in allWords if word not in self.stopWordDict]
        
        wordCount = Counter(subWords)  # 统计词频，排序
        sortWordCount = sorted(wordCount.items(), key=lambda x: x[1], reverse=True)
        
        # 去除低频词
        words = [item[0] for item in sortWordCount if item[1] >= self.miniFreq ]
        
        #获取词列表和顺序对应的预训练权重矩阵
        vocab, wordEmbedding = self.getWordEmbedding(words)
        
        self.wordEmbedding = wordEmbedding
        
        self.wordToIndex = dict(zip(vocab, list(range(len(vocab)))))
        self.indexToWord = dict(zip(list(range(len(vocab))), vocab))
        self.n_symbols = len(self.wordToIndex) + 1
        
        # 将词汇-索引映射表保存为json数据，之后做inference时直接加载来处理数据
        with open("../data/wordJson/wordToIndex.json", "w", encoding="utf-8") as f:
            json.dump(self.wordToIndex, f)
        
        with open("../data/wordJson/indexToWord.json", "w", encoding="utf-8") as f:
            json.dump(self.indexToWord, f)
    
    def reviewProcess(self, review, sequenceLength, wordToIndex):
        """
        将数据集中的每条评论里面的词，根据词表，映射为index表示
        每条评论 用index组成的定长数组来表示
        """
        reviewVec = np.zeros((sequenceLength))
        sequenceLen = sequenceLength
        
        # 判断当前的序列是否小于定义的固定序列长度
        if len(review) < sequenceLength:
            sequenceLen = len(review)
            
        for i in range(sequenceLen):
            if review[i] in wordToIndex:
                reviewVec[i] = wordToIndex[review[i]]
            else:
                reviewVec[i] = wordToIndex["UNK"]

        return reviewVec
    
    def genTrainEvalData(self, x, y, rate):
        """
        生成训练集和验证集
        """
        reviews = []
        labels = []
        
        # 遍历所有的文本，将文本中的词转换成index表示
        for i in range(len(x)):
            reviewVec = self.reviewProcess(x[i], self.sequenceLength, self.wordToIndex)
            reviews.append(reviewVec)
            labels.append([y[i]])
            
        trainIndex = int(len(x) * rate)
       
        #trainReviews = sequence.pad_sequences(reviews[:trainIndex], maxlen=self.sequenceLength)
        trainReviews = np.asarray(reviews[:trainIndex], dtype="int64")
        trainLabels = np.array(labels[:trainIndex], dtype="float32")
        trainLabels = to_categorical(trainLabels,num_classes=2) 
        
        #evalReviews = sequence.pad_sequences(reviews[trainIndex:], maxlen=self.sequenceLength)
        evalReviews = np.asarray(reviews[trainIndex:], dtype="int64")
        evalLabels = np.array(labels[trainIndex:], dtype="float32")
        evalLabels = to_categorical(evalLabels,num_classes=2) 

        return trainReviews, trainLabels, evalReviews, evalLabels
        
    def dataGen(self):
        """
        初始化训练集和验证集
        """
        
        #读取停用词
        self.readStopWord(self.stopWordSource)
        
        #读取数据集
        reviews, labels = self.readData(self.dataSource)
        
        #分词、去停用词
        #生成 词汇-索引 映射表和预训练权重矩阵，并保存
        self.genVocabulary(reviews)
        
        
        #初始化训练集和测试集
        trainReviews, trainLabels, evalReviews, evalLabels = self.genTrainEvalData(reviews, labels, self.rate)
        self.trainReviews = trainReviews
        self.trainLabels = trainLabels
        
        self.evalReviews = evalReviews
        self.evalLabels = evalLabels
        
        
data = Dataset(config)
data.dataGen()

class Position_Embedding(Layer):

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size #必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size,seq_len = K.shape(x)[0],K.shape(x)[1]
        position_j = 1. / K.pow(10000., \
                                 2 * K.arange(self.size / 2, dtype='float32' \
                               ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:,:,0]), 1)-1 #K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2]+self.size)


class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:,0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape)-2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        #如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        #如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq,K_seq,V_seq = x
            Q_len,V_len = None,None
        elif len(x) == 5:
            Q_seq,K_seq,V_seq,Q_len,V_len = x
        #对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0,2,1,3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0,2,1,3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0,2,1,3))
        #计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3,3]) / self.size_per_head**0.5
        A = K.permute_dimensions(A, (0,3,2,1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0,3,2,1))
        A = K.softmax(A)
        #输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3,2])
        O_seq = K.permute_dimensions(O_seq, (0,2,1,3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

def transfromer(n_symbols,embedding_weights,config):
    S_inputs = Input(shape=(None,), dtype='int32')

    embeddings =Embedding(input_dim=n_symbols, output_dim=config.embeddingSize,
                            weights=[embedding_weights],
                            input_length=config.sequenceLength)(S_inputs)

    #增加Position_Embedding能轻微提高准确率
    embeddings = Position_Embedding()(embeddings) 

    O_seq = Attention(8,16)([embeddings,embeddings,embeddings])

    O_seq = GlobalAveragePooling1D()(O_seq)

    O_seq = Dropout(config.dropoutKeepProb)(O_seq)

    outputs = Dense(2, activation='softmax')(O_seq)

    model = Model(inputs=S_inputs, outputs=outputs)
    model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    return model

wordEmbedding = data.wordEmbedding
n_symbols=data.n_symbols
model = transfromer(n_symbols,wordEmbedding,config)
model.summary()

x_train = data.trainReviews
y_train = data.trainLabels
x_eval = data.evalReviews
y_eval = data.evalLabels

wordEmbedding = data.wordEmbedding
n_symbols=data.n_symbols

reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=10, mode='auto')

early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model_checkpoint = ModelCheckpoint('./transfromer_model/best_model/model_{epoch:02d}-{val_accuracy:.2f}.hdf5', save_best_only=True, save_weights_only=True)
history = model.fit(x_train, y_train, batch_size=config.batchSize, epochs=config.epochs, validation_split=0.3,shuffle=True, callbacks=[reduce_lr,early_stopping,model_checkpoint])
#验证

scores = model.evaluate(x_eval, y_eval)

#保存模型
yaml_string = model.to_yaml()
with open('./transfromer_model/transfromer.yml', 'w') as outfile:
    outfile.write( yaml.dump(yaml_string, default_flow_style=True) )
model.save_weights('./transfromer_model/transfromer.h5')

print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))

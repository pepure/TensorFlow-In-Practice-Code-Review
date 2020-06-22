#!/usr/bin/env python
# coding: utf-8

# 这是一个比较有趣的自动生成《莎士比亚十四行诗》风格诗的代码。

# 1. 载入一些库

# In[12]:


from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional,Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np


# 2. 读取《莎士比亚十四行诗》并统一转换成小写

# In[13]:


data = open("tmp/sonnets.txt").read()
corpus = data.lower().split("\n")


# 3. 将词转换成序列，Tokenizer生成了一个字典，统计词频等信息

# In[14]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
#print(total_words)


# 4. 生成输入数据序列

# In[15]:


input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i + 1]
        input_sequences.append(n_gram_sequence)
#print(input_sequences)


# 5. 按照最大的序列长度进行序列的前补齐

# In[16]:


max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))
#print(input_sequences)


# 6. 创建训练数据集和标签

# In[17]:


predictors, label = input_sequences[:, :-1], input_sequences[:, -1]
#print(predictors)
#print(label)


# 7. 将标签转换为One-Hot的形式

# In[18]:


label = ku.to_categorical(label, num_classes=total_words)
#print(label.shape)
#print(label)


# 8. 搭建模型并编译模型

# 把整数序列用 one-hot 编码转成高维稀疏向量再转成低维密集向量，因此一个词可能用 100D, 300D 或 500D 的向量来表示，而词与词之间的相似度就用向量之间内积表示。这个过程就叫做 word2vec，即把单词转成换向量的过程，在 keras 里用 Embedding() 来实现。
# ![image.png](attachment:image.png)

# In[19]:


model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len - 1))
model.add(Bidirectional(LSTM(150, return_sequences=True)))
model.add(Dropout(0.2))
model.add(LSTM(100))
model.add(Dense(total_words / 2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())


# 9. 开始训练

# In[20]:


history = model.fit(predictors, label, epochs=100, verbose=1)


# 10. 查看训练的曲线

# In[21]:


import matplotlib.pyplot as plt

acc = history.history['accuracy']
loss = history.history['loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')

plt.figure()

plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()

plt.show()


# 保存模型，演示的时候可以直接载入进行测试

# In[22]:


#model.save("tmp/L3.h5")
#model = load_model("tmp/L3.h5")
#print(model.summary())


# 11. 自动生成莎士比亚风格的诗句

# In[23]:


seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100
#print(max_sequence_len)
for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    #print(token_list)
    token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='pre')
    #print(token_list)
    predicted = model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)


# 帮助我奥比万克诺比，你是我唯一的希望，在他的道路上，某种染成了证明你年轻的时候，独自骑着新的富有，有这样的哑巴，我的所有的心思，都是青春的交叉时间，你是邪恶的旧的，我的行为向你弯曲，你释放你，你一个人，我的爱，所以说，我的眼睛是‘意志’，那里，所以它明亮的愤怒，仍然带着光明的愤怒，活着，白色的新信仰，打电话给我，所以认为，敌人的胸部变化，做一些不好的东西，新的自由，旧的变化，和你在一起 

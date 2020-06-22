#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
print(tf.__version__)


# 1. 载入fashion-mnist数据集

# In[2]:


mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()


# 2. 数据集维度调整 → 归一化

# In[3]:


training_images=training_images.reshape(60000, 28, 28, 1)
training_images=training_images / 255.0
test_images = test_images.reshape(10000, 28, 28, 1)
test_images=test_images/255.0


# 3. 卷积神经网络（7层）模型搭建

# In[4]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
  tf.keras.layers.MaxPooling2D(2, 2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])


# 4. 模型编译并打印确认

# In[5]:


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()


# 5. 模型训练 5epochs

# In[6]:


model.fit(training_images, training_labels, epochs=5)


# 6. 模型评估

# In[7]:


print(model.metrics_names)
test_result = model.evaluate(test_images, test_labels)
print(test_result)


# 7. 模型预测

# In[8]:


import matplotlib.pyplot as plt
print(test_labels[10])
test_predict = model.predict(test_images[10].reshape(1, 28, 28, 1))
print(test_predict)
test_class = np.argmax (test_predict)
print(test_class)
plt.imshow(test_images[10].reshape(28, 28))


# 8. 好玩的内容：输出CNN网络中间层学习到的内容

# In[13]:


f, axarr = plt.subplots(3,4)
FIRST_IMAGE=1
SECOND_IMAGE=2
THIRD_IMAGE=3
CONVOLUTION_NUMBER = 30 #第2个核学到的
from tensorflow.keras import models
layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(inputs = model.input, outputs = layer_outputs)
for x in range(0,4):#2个卷积和2个MaxPool
  f1 = activation_model.predict(test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[0,x].imshow(f1[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[0,x].grid(False)
  f2 = activation_model.predict(test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[1,x].imshow(f2[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[1,x].grid(False)
  f3 = activation_model.predict(test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x]
  axarr[2,x].imshow(f3[0, : , :, CONVOLUTION_NUMBER], cmap='inferno')
  axarr[2,x].grid(False)


# 答案揭晓：

# In[14]:


plt.imshow(test_images[2].reshape(28, 28))


# In[2]:


get_ipython().system('jupyter nbconvert --to python file_name.ipynb')


# 9. End

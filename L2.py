#!/usr/bin/env python
# coding: utf-8

# 本案例主要使用InceptionV3模型进行迁移学习，来进行"人和马训练集"的分类。

# 1. 导入类库，包括keras自带的InceptionV3模型

# In[1]:


import os
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Model
from os import getcwd

from tensorflow.keras.applications.inception_v3 import InceptionV3


# 2. 载入InceptionV3模型
# include_top = False，不包含最后一层全连接层

# In[2]:


pre_trained_model = InceptionV3(input_shape = (150, 150, 3), 
                                include_top = False, 
                                weights = None)


# 3. 导入预训练的参数(李飞飞ImageNet训练集)

# In[3]:


local_weights_file = "tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
pre_trained_model.load_weights(local_weights_file)


# 4. 冻结所有参数，不参与新的训练集的训练

# In[4]:


for layer in pre_trained_model.layers:
    layer.trainable = False


# 5. 打印预训练模型的结构
# 我们可以看到，InceptionV3是有个很深的网络模型，得益于巧妙的Inception结构，使得该模型的梯度不会消失

# In[5]:


pre_trained_model.summary()


# 6. 基于预训练模型，搭建新适合的模型
# 这里最后重新增加全连接层，替换原来去掉的全连接层

# In[6]:


last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output

# Flatten the output layer to 1 dimension
x = layers.Flatten()(last_output)
# Add a fully connected layer with 1,024 hidden units and ReLU activation
x = layers.Dense(1024, activation='relu')(x)
# Add a dropout rate of 0.2
x = layers.Dropout(0.2)(x)                  
# Add a final sigmoid layer for classification
x = layers.Dense(1, activation='sigmoid')(x)
model = Model( pre_trained_model.input, x) 


# 7. 编译模型并查看新模型的结构

# In[7]:


from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer = RMSprop(lr=0.0001), 
              loss = 'binary_crossentropy', 
              metrics = ['accuracy'])
model.summary()


# 8. 载入本地的"人和马数据集"
# 同时对训练集进行一定的数据增强（训练时随机变换达到的增强效果，并非增加数据集数量）

# In[8]:


path_horse_or_human = "tmp/horse-or-human.zip"
path_validation_horse_or_human = "tmp/validation-horse-or-human.zip"

import os
import zipfile
import shutil

if not(os.path.exists("tmp/training")):
    #shutil.rmtree("tmp")
    local_zip = path_horse_or_human
    zip_ref = zipfile.ZipFile(local_zip, "r")
    zip_ref.extractall("tmp/training")
    zip_ref.close()

    local_zip = path_validation_horse_or_human
    zip_ref = zipfile.ZipFile(local_zip, "r")
    zip_ref.extractall("tmp/validation")
    zip_ref.close()
    
train_dir = "tmp/training"
validation_dir = "tmp/validation"

train_horses_dir = os.path.join(train_dir, "horses")
train_humans_dir = os.path.join(train_dir, "humans")
validation_horses_dir = os.path.join(validation_dir, "horses")
validation_humans_dir = os.path.join(validation_dir, "humans")

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)

print(len(train_horses_fnames))
print(len(train_humans_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))


# In[9]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255.,
                                   rotation_range = 40,
                                   width_shift_range = 0.2,
                                   height_shift_range = 0.2,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1.0/255.)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    batch_size = 20,
                                                    class_mode = 'binary', 
                                                    target_size = (150, 150))   

validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                          batch_size  = 20,
                                                          class_mode  = 'binary',     
                                                         target_size = (150, 150))


# 9. 增加训练终止的条件

# In[10]:


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('accuracy')>0.99):
            print("\nReached 99.0% accuracy so cancelling training!")
            self.model.stop_training = True
            
callbacks = myCallback()


# 10. 终于，正式开始训练！

# In[ ]:


history = model.fit_generator(train_generator,
            validation_data = validation_generator,
            steps_per_epoch = 50,
            epochs = 3,
            validation_steps = 12,
            verbose = 1,
            callbacks=[callbacks])


# 11. 显示训练的accuracy曲线

# In[13]:


import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()

plt.show()


# 因为相对于InceptionV3模型来说，该数据集很小，只是作为测试，从这个数据上看，应该是有些过拟合。

# - End -

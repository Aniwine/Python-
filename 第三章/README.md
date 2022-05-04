## 第五章 深度学习用于计算机视觉

### 5.0 概览

主要内容：

1. 理解卷积神经网络（convnet）  
2. 使用数据增强来降低过拟合  
3. 使用预训练的卷积神经网络进行特征提取  
4. 微调预训练的卷积神经网络  
5. 将卷积神经网络学到的内容及其如何做出分类决策可视化  

### 5.1 卷积神经网络简介

先实例化一个简单的卷积神经网络模型,并做出简单的说明  

```python
from keras import layers
from keras import models
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
```

可以看到卷积神经网络是conv2D层和MaxPooling2D层的交替堆叠。  

卷积神经网络的接受形状是(image_height,image_width,image_channels),这是一个3D张量，从第二章可以知道图像一般存储与4D张量中，但是不管是深度神经网络还是卷积神经网络，我们接受的形状都只需要特征shape，那么自然这里也不包括样本维度（或批量维度）。**对于整个卷积神经网络的接受形状只需要在第一层传入input_shape=(image_height,image_width,image_channels)**即可，本例中即(28,28,1)。

列出卷积神经网络的架构

```python
>>> model.summary()
_________________________________________________________________
Layer (type) Output Shape 								Param #
=================================================================
conv2d_1 (Conv2D) (None, 26, 26, 32) 					320
_________________________________________________________________
max_pooling2d_1 (MaxPooling2D) (None, 13, 13, 32) 		0
_________________________________________________________________
conv2d_2 (Conv2D) (None, 11, 11, 64) 					18496
_________________________________________________________________
max_pooling2d_2 (MaxPooling2D) (None, 5, 5, 64) 		0
_________________________________________________________________
conv2d_3 (Conv2D) (None, 3, 3, 64) 						36928
=================================================================
Total params: 55,744
Trainable params: 55,744
Non-trainable params: 0
```

> 回忆一下神经网络，第一层输入的input_shape为(28,28).reshape(-1,1),即将所有特征列展平，然后作为这一层权重矩阵的行数，而列数就是这一层输出的神经元个数(new_features)，那么这一层输出矩阵就是  （samples,new_features),因为我们可以很清晰的知道矩阵运算，输入矩阵的一行（就是一个样本）乘以该层的权重矩阵的一列，所以得出权重矩阵的行数就应该是输入矩阵的列数（即输入了多少个特征）。  
>
> 但是在卷积神经网络里，我们无法想象每一层做的运算是怎么样的，也就很难计算该层该有多少个参数，以及输入、输出（不再是矩阵）代表着什么，又是多少？带着这个疑问继续向后看。

虽然暂时还无法理解，但是可以知道，我们可以设置输入的形状（上面提到），每一层的输出也可以设置

> 对于Conv2D层和MaxPooling2D层而言，**输出为(height,width,channels)**的一个3D张量。但是请注意，设定这些参数时**顺序不一样，他将通道数量放在第一个，而宽高作为元组放在第二个上，即(channels,(height,width)**,并且通道数量一般设置为32或64。这三个维度的意义待会再讲。

现在需要将输出张量输入到神经网络里(密集连接分类网络，就是第二章那个)，因为是接上面model添加的层，所以直接使用Flatten()就会将输入的特征张量展平为一层（不包括样本维度）  

```python
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
```

查看网络架构

```python
>>> model.summary()
_________________________________________________________________  
Layer (type) Output Shape 								Param #  
=================================================================   
conv2d_1 (Conv2D) (None, 26, 26, 32) 					320  
_________________________________________________________________
max_pooling2d_1 (MaxPooling2D) (None, 13, 13, 32) 		0  
_________________________________________________________________  
conv2d_2 (Conv2D) (None, 11, 11, 64) 					18496  
_________________________________________________________________  
max_pooling2d_2 (MaxPooling2D) (None, 5, 5, 64) 		0  
_________________________________________________________________  
conv2d_3 (Conv2D) (None, 3, 3, 64) 						36928  
_________________________________________________________________  
flatten_1 (Flatten) (None, 576) 						0  
_________________________________________________________________  
dense_1 (Dense) (None, 64) 								36928   
_________________________________________________________________  
dense_2 (Dense) (None, 10) 								650  
=================================================================  
Total params: 93,322
Trainable params: 93,322
Non-trainable params: 0
```

下面训练并测试，可以得出准确率非常高，为99.3%

```python
from keras.datasets import mnist
from keras.utils import to_categorical
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
我们在测试数据上对模型进行评估。
>>> test_loss, test_acc = model.evaluate(test_images, test_labels)
>>> test_acc
0.99080000000000001
```


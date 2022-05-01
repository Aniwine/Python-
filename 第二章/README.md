## 第二章 神经网络的数学基础
### 2.0 概览 
> 张量，张量运算，微分，梯度下降，反向传播等
>
> 注：文中所写的代码只是笔者个人觉得要注意。并不全，需要完整代码的请参考书籍。

### 2.1 初识神经网络
#### 1.数据集
mnist数据集：将手写数字的灰度图像（28\*28像素）划分到10个类别里。  
测试集包含60000训练图像和10000张测试图像    

```python
from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
```



**类和标签的说明**
在分类问题中某个类别叫做类（class)，数据点叫做样本（sample）,某个样本对应的类叫做标签（label）

> 此数据集加载数据时进行分割(train_images,train_labels),(test_images,test_labels)=mnist.load_data()

#### 2.网络架构

**核心是构造层和编译**   

**层(layer)**,是一种数据处理模块，可以将他看成一个数据过滤器，进去一些数据，出来的数据更加有用。而神经网络就是有很多这样的层结构组成的数据处理的“筛子”   

本例中使用两个Dense层，也叫全连接层，最后一层为softmax层，是输出层，作用是将原本的输出值转换为概率值（总和为1），每一个概率表示分属于每一个类别的概率。本例有十个类别。 

```python
from keras import models
from keras import layers
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
```

  

**编译的三个重要参数**

- 损失函数（loss function):网络衡量训练数据性能的依据。即这次训练的结果好不好，有多好。  
- 优化器（optimizer）：根据训练数据和损失函数来更新网络的机制。(更新网络即调整参数)
- 训练、测试需要监控的指标（metric）：本例只指定了精度（accuracy）  

```python
network.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
```

#### 3.数据预处理

神经网络每一层的数据要求必须是二维的，即（samples,features），即需要把多维的特征占平成一维。至于是否需要标准化或者其他处理，看实际情况。  最后需要对标签进行分类编码。

> 本例将[0,255]的取值范围收缩在[0,1]范围之内。

```pytho
# 标签分类编码
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```


"""
    用 mnist举例 keras实现分类
"""
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils # keras自带的一些np脚本
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import RMSprop

# download the mnist to the path '~/.keras/datasets/' if it is the first time to be called
# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(X_train.shape[0], -1) / 255.      # normalize
X_test = X_test.reshape(X_test.shape[0], -1) / 255.         # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)  # 把 0-9 的标签转换成 one-hot形式
y_test = np_utils.to_categorical(y_test, num_classes=10)

# Another way to build your neural net
# 这和 pytorch也忒像了吧
model = Sequential([
    Dense(32 , input_dim=28*28),
    Activation('relu'),
    Dense(10),
    Activation('softmax'), # 2选1用sigmoid,多选一用softmax
])


# Another way to define your optimizer
rmsprop = RMSprop(lr = 0.001 , rho=0.9 , epsilon=1e-08 , decay = 0.0)


model.compile(optimizer=rmsprop,
              loss='categorical_crossentropy',  # one-hot版的交叉熵
              metrics=['accuracy']              # 训练时每次显示准确率
              )

print('Training ------------')
# 新的训练方法:model.fit
model.fit(X_train, y_train, epochs=20, batch_size=32)

print('\nTesting ------------')
# Evaluate the model with the metrics we defined earlier
loss, accuracy = model.evaluate(X_test, y_test)  # evaluate返回compile里规定的参数:loss,accuracy

print('test loss: ', loss)
print('test accuracy: ', accuracy)  # epochs = 2 -> 95% , epochs = 5 -> 96% , epochs = 20 -> 96.6%
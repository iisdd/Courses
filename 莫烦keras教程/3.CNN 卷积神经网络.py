"""
    手写数字举例说明keras如何实现CNN,网络结构:两层CNN、两层FC,训练起来血慢
"""
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D, MaxPooling2D, Flatten
# 这个import看起来有点搞笑啊,买零件一样...
from keras.optimizers import Adam

# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing,归一化处理,0-9转化成 one-hot
X_train = X_train.reshape(-1, 1,28, 28)/255.
X_test = X_test.reshape(-1, 1,28, 28)/255.
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)

# 建立CNN
model = Sequential()

# 第一层卷积层 (1,28,28) -> (32,28,28)
model.add(Convolution2D(
    batch_input_shape=(None , 1 , 28 , 28),
    filters=32,                     # 把特征卷成32个
    kernel_size=5,
    strides=1,
    padding='same',
    data_format='channels_first',   # 第一个放厚度
))
model.add(Activation('relu'))

# 第一层池化层(32,28,28) -> (32,14,14)
model.add(MaxPooling2D(
    pool_size=2,
    strides=2,
    padding='same',
    data_format='channels_first',
))

# 第二层卷积层(32,14,14) -> (64,14,14)
model.add(Convolution2D(64,5,strides=1,padding='same',data_format='channels_first'))
model.add(Activation('relu'))

# 第二层池化层(64,14,14) -> (64,7,7)
model.add(MaxPooling2D(2,2,padding='same',data_format='channels_first'))

# 第一层全连接(64,7,7) -> (1024)
model.add(Flatten())  # (64,7,7) -> (3136),抹平成一维
model.add(Dense(1024))
model.add(Activation('relu'))

# 第二层全连接
model.add(Dense(10))
model.add(Activation('softmax'))

adam = Adam(lr = 1e-4)

model.compile(
    optimizer=adam,
    loss='categorical_crossentropy',
    metrics=['accuracy'] # metrics:度量
)

print('=========trianing=========')
model.fit(X_train , y_train , epochs = 2 , batch_size=64 , verbose=2) # verbose = 2:每一个epoch打印一次进度

print('\n==========testing=========')
loss, accuracy = model.evaluate(X_test , y_test)

print('\ntest loss: ', loss)
print('\ntest accuracy: ', accuracy)  # epochs = 1 -> 0.968 牛! , epochs = 3 -> 0.986 , epochs = 20 -> 99.2%
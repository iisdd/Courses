"""
    用手写数字的例子展示 keras普通 RNN的用法
"""
import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.utils import np_utils   # 一些 keras自带的,用 numpy写的脚本
from keras.models import Sequential
from keras.layers import SimpleRNN , Activation , Dense
from keras.optimizers import Adam

TIME_STEPS = 28 # 相当于图片的高度
INPUT_SIZE = 28 # 图片的宽度,每次送进去的 features
BATCH_SIZE = 50
BATCH_INDEX = 0
OUTPUT_SIZE = 10
CELL_SIZE = 50  # RNN输出50个神经元,再接一个全连接(50 , 10)
LR = 0.001


# training X shape (60000, 28x28), Y shape (60000, ). test X shape (10000, 28x28), Y shape (10000, )
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# data pre-processing
X_train = X_train.reshape(-1, 28, 28) / 255.      # normalize
X_test = X_test.reshape(-1, 28, 28) / 255.        # normalize
y_train = np_utils.to_categorical(y_train, num_classes=10)
y_test = np_utils.to_categorical(y_test, num_classes=10)


# 建立RNN网络
model = Sequential()

# 先送进RNN
model.add(SimpleRNN(
    input_shape=(TIME_STEPS , INPUT_SIZE),
    output_dim = CELL_SIZE,
    unroll=True
))

# 接一层全连接,多种类分类用softmax激活
model.add(Dense(OUTPUT_SIZE))
model.add(Activation('softmax'))

# 编译
adam = Adam(LR)
model.compile(optimizer=adam,
              loss = 'categorical_crossentropy',
              metrics=['accuracy']) # model.evaluate时显示 loss 和 accuracy

# 手动抽样训练
for step in range(4001):
    # X有三维(samples , time_steps , features)
    X_batch = X_train[BATCH_INDEX : BATCH_INDEX + BATCH_SIZE , : , : ]
    Y_batch = y_train[BATCH_INDEX : BATCH_INDEX + BATCH_SIZE , : ]
    cost = model.train_on_batch(X_batch , Y_batch)
    BATCH_INDEX += BATCH_SIZE
    BATCH_INDEX = 0 if BATCH_INDEX >= X_train.shape[0] else BATCH_INDEX

    if step % 500 == 0:
        cost , accuracy = model.evaluate(X_test , y_test , batch_size=y_test.shape[0] , verbose=False)
        print('test cost : ' , cost , 'test accuracy : ' , accuracy) # 93.6%

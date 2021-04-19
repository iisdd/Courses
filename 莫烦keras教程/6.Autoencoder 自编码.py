"""
    图片的压缩与解压,展现 keras的自编码
"""
import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
from keras.models import Model  # 新东西
from keras.layers import Dense , Input  # 又是新东西
import matplotlib.pyplot as plt

(x_train , _) , (x_test , y_test) = mnist.load_data()  # 自编码不需要train_y

x_train = x_train.astype('float32') / 255. - 0.5       # minmax_normalized
x_test = x_test.astype('float32') / 255. - 0.5         # minmax_normalized
x_train = x_train.reshape((x_train.shape[0], -1))
x_test = x_test.reshape((x_test.shape[0], -1))
# print(x_train.shape) # (60000, 784)
# print(x_test.shape) # (10000, 784)

encoding_dim = 2 # 把特征压到两个,方便画图

# 设置传入值
input_img = Input(shape=(28*28 , ))

# 编码层,新写法我日
encoded = Dense(128 , activation='relu')(input_img)
encoded = Dense(64 , activation='relu')(encoded)
encoded = Dense(10 , activation='relu')(encoded)
encoder_output = Dense(encoding_dim)(encoded)

# 解码层
decoded = Dense(10 , activation='relu')(encoder_output)
decoded = Dense(64 , activation='relu')(decoded)
decoded = Dense(128 , activation='relu')(decoded)
decoded = Dense(28*28 , activation='tanh')(decoded)  # tanh恢复到 -1 ~ 1

# 自定义autoencoder模型
autoencoder = Model(input = input_img , output = decoded)

# 自定义encoder部分
encoder = Model(input = input_img , output = encoder_output)

autoencoder.compile(loss='mse' , optimizer='adam')

# 训练
autoencoder.fit(x=x_train,y=x_train,epochs=20,batch_size=256,shuffle=True,verbose=2)
# train_x:(60000, 784)

# 画图
encoded_imgs = encoder.predict(x_test)
plt.scatter(encoded_imgs[:, 0], encoded_imgs[:, 1], c=y_test) # 按 y_test涂不同的颜色
plt.colorbar()
plt.show()

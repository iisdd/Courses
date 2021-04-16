"""
    展示一下几种 activation function 的图形
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# fake data
x = np.linspace(-5, 5, 200)     # x data, shape=(100, 1)
x2 = np.linspace(0, 1, 5)       # 给softmax用的
# following are popular activation functions
y_relu = tf.nn.relu(x)
y_sigmoid = tf.nn.sigmoid(x)
y_tanh = tf.nn.tanh(x)
y_softplus = tf.nn.softplus(x)
y_softmax = tf.nn.softmax(x2)   # softmax is a special kind of activation function, it is about probability

sess = tf.Session()
# 输出也是一条长的向量
y_relu, y_sigmoid, y_tanh, y_softplus, y_softmax = sess.run([y_relu, y_sigmoid, y_tanh, y_softplus, y_softmax])

# plt to visualize these activation function
plt.figure(1, figsize=(8, 6))
plt.subplot(231)
plt.plot(x, y_relu, c='red', label='relu')
plt.ylim((-1, 5))
plt.legend(loc='best')

plt.subplot(232)
plt.plot(x, y_sigmoid, c='red', label='sigmoid')
plt.ylim((-0.2, 1.2))
plt.legend(loc='best')

plt.subplot(233)
plt.plot(x, y_tanh, c='red', label='tanh')
plt.ylim((-1.2, 1.2))
plt.legend(loc='best')

plt.subplot(234)
plt.plot(x, y_softplus, c='red', label='softplus')
plt.ylim((-0.2, 6))
plt.legend(loc='best')

plt.subplot(235)
plt.plot(x2, y_softmax, c='red', label='softmax')
plt.ylim((0, 0.4))
plt.legend(loc='best')


plt.show()
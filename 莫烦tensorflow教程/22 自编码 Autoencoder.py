'''
    encoder压缩手写数字图片,然后decoder解压,压缩过程有点像降维(PCA),非监督学习的分类问题
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # 用来画压缩后的 3D特征图
from matplotlib import cm
import numpy as np

tf.set_random_seed(1)

# 超参数
BATCH_SIZE = 64
LR = 0.002
N_TEST_IMG = 5

# 加载数据
mnist = input_data.read_data_sets('./MNIST_data' , one_hot = False)  # 输出 0-9 而不是长条的 010101
test_x = mnist.test.images[ : 200]
test_y = mnist.test.labels[ : 200]

# # 画个例子
# print(mnist.train.images.shape)     # (55000, 28 * 28)
# print(mnist.train.labels.shape)     # (55000, 10)
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
# plt.title('%i' % np.argmax(mnist.train.labels[0]))
# plt.show()

# 设置传入值
tf_x = tf.placeholder(tf.float32 , [None , 28 * 28])

# encoder 部分 , 把 28*28 个特征压到 3 个特征
en0 = tf.layers.dense(tf_x , 128 , tf.nn.tanh)
en1 = tf.layers.dense(en0 , 64 , tf.nn.tanh)
en2 = tf.layers.dense(en1 , 12 , tf.nn.tanh)
encoded = tf.layers.dense(en2 , 3)

# decoder 部分 , 把 3 个特征展开成 28 * 28个特征
de0 = tf.layers.dense(encoded , 12 , tf.nn.tanh)
de1 = tf.layers.dense(de0 , 64 , tf.nn.tanh)
de2 = tf.layers.dense(de1 , 128 , tf.nn.tanh)
decoded = tf.layers.dense(de2 , 28*28 , tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x , predictions=decoded)
train = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# 初始化图片框
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))  # f:figure, a:axis
plt.ion()   # continuously plot

# 选前5个数字展示,画在左边
view_data = mnist.test.images[:N_TEST_IMG]
for i in range(N_TEST_IMG): # 在第一行画原图
    a[0][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
    a[0][i].set_xticks(()); a[0][i].set_yticks(())


# 训练网络部分
for step in range(5000):
    b_x , b_y = mnist.train.next_batch(BATCH_SIZE)
    # _ , encoded_ , decoded_ , loss_ = sess.run([train , encoded , decoded , loss] , {tf_x : b_x})
    _, loss_ = sess.run([train, loss], {tf_x: b_x})     # 反正encoded_, decoded_也不会取出来用
    if step % 100 == 0:
        print('train loss : %.4f' % loss_)
        # 在第二行画 decoded结果图
        decoded_data = sess.run(decoded , {tf_x : view_data})
        for i in range(N_TEST_IMG):
            a[1][i].clear()
            a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(()); a[1][i].set_yticks(())
        plt.draw(); plt.pause(0.01)
plt.ioff()

# 3D特征图
view_data = test_x[:200]
encoded_data = sess.run(encoded, {tf_x: view_data})
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
for x, y, z, s in zip(X, Y, Z, test_y):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()


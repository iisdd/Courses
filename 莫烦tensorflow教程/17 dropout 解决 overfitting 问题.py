"""
    本节用一个曲线拟合的例子体现普通训练的 overfitting以及 dropout 的优越性
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# 超参数
N_SAMPLES = 20
N_HIDDEN = 300      # 大网络,过拟合
LR = 0.01

# training data
x = np.linspace(-1 , 1 , N_SAMPLES)[ : ,np.newaxis]
y = x + 0.3 * np.random.randn(N_SAMPLES)[ : ,np.newaxis]  # 加白噪声

# test data
test_x = x.copy()
test_y = test_x + 0.3 * np.random.randn(N_SAMPLES)[ : ,np.newaxis]

# 展示数据
plt.scatter(x, y, c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# 设置传入值
tf_x = tf.placeholder(tf.float32 , [None , 1])
tf_y = tf.placeholder(tf.float32 , [None , 1])
tf_is_training = tf.placeholder(tf.bool , None)                         # 训练时就有dropout,否则就没有

# 搭建overfitting网络:
# placeholder - 网络层 - 输出层 - loss - train_op
o1 = tf.layers.dense(tf_x , N_HIDDEN , tf.nn.relu)
o2 = tf.layers.dense(o1 , N_HIDDEN , tf.nn.relu)
o_out = tf.layers.dense(o2 , 1)
o_loss = tf.losses.mean_squared_error(tf_y , o_out)
o_train = tf.train.AdamOptimizer(LR).minimize(o_loss)

# 搭建dropout网络,就是每层中间夹一层dropout
d1 = tf.layers.dense(tf_x , N_HIDDEN , tf.nn.relu)
d1 = tf.layers.dropout(d1 , rate = 0.5 , training = tf_is_training)     # dropout掉 50%的神经元
d2 = tf.layers.dense(d1 , N_HIDDEN , tf.nn.relu)
d2 = tf.layers.dropout(d2 , rate = 0.5 , training = tf_is_training)
d_out = tf.layers.dense(d2 , 1)
d_loss = tf.losses.mean_squared_error(tf_y , d_out)
d_train = tf.train.AdamOptimizer(LR).minimize(d_loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()

for i in range(500):
    sess.run([o_train , d_train] , feed_dict={tf_x : x , tf_y : y , tf_is_training : True})

    if i % 10 == 0:
        plt.cla()                                                       # 清除轴
        o_loss_ , d_loss_ , o_out_ , d_out_ = sess.run(
            [o_loss , d_loss , o_out , d_out],
            {tf_x : test_x , tf_y : test_y , tf_is_training : False},   # 测试的时候不需要dropout
        )

        # 画图部分,直接复制了
        plt.scatter(x, y, c='magenta', s=50, alpha=0.3, label='train'); plt.scatter(test_x, test_y, c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x, o_out_, 'r-', lw=3, label='overfitting'); plt.plot(test_x, d_out_, 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % o_loss_, fontdict={'size': 20, 'color':  'red'}); plt.text(0, -1.5, 'dropout loss=%.4f' % d_loss_, fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left'); plt.ylim((-2.5, 2.5)); plt.pause(0.1)

plt.ioff()
plt.show()

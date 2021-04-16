# 本节展示不同优化器的 cost 优化效果,例子是曲线拟合

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

tf.set_random_seed(1)
np.random.seed(1)

LR = 0.01
BATCH_SIZE = 32

# 创建假数据
x = np.linspace(-1 , 1 , 100)[ : ,np.newaxis]                       # shape(100 , 1)
noise = np.random.normal(0 , 0.1 , size = x.shape)
y = np.power(x , 2) + noise

# plot dataset
# plt.scatter(x , y)  # 散点图
# plt.show()

# 创建神经网络
class Net:
    def __init__(self , opt , **kwargs):                            # **kwargs 代表不同的优化器输入不同的参数
        self.x = tf.placeholder(tf.float32 , [None , 1] )
        self.y = tf.placeholder(tf.float32 , [None , 1] )
        l1 =  tf.layers.dense(self.x , 20 , tf.nn.relu)
        out = tf.layers.dense(l1 , 1)
        self.loss = tf.losses.mean_squared_error(self.y , out)
        self.train = opt(LR , **kwargs).minimize(self.loss)         # 有的优化器有别的参数,比如 momentum

# 创建四个不同优化器的网络
net_SGD = Net(tf.train.GradientDescentOptimizer)
net_Momentum = Net(tf.train.MomentumOptimizer , momentum = 0.9)
net_RMSprop = Net(tf.train.RMSPropOptimizer)
net_Adam = Net(tf.train.AdamOptimizer)
nets = [net_SGD , net_Momentum , net_RMSprop , net_Adam]

# 激活起手式
sess = tf.Session()
sess.run(tf.global_variables_initializer())

losses_his = [[], [], [], []]

# 抽样训练
for step in range(300):
    index = np.random.randint(0 , x.shape[0] , BATCH_SIZE)  # 抽 32个点
    b_x = x[index]
    b_y = y[index]

    for net, l_his in zip(nets , losses_his):
        _ , l = sess.run([net.train , net.loss] , feed_dict={net.x : b_x , net.y : b_y})
        l_his.append(l)

# plot loss history
labels = ['SGD' , 'Momentum' , 'RMSprop' , 'Adam']
for i,l_his in enumerate(losses_his):
    plt.plot(l_his , label = labels[i])     # label= 规定了不同颜色线的名称
plt.legend(loc = 'best')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.ylim(0 , 0.2)
plt.show()

# 太顶了吧 Adam
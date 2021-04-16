'''
用两个神经网络,一个带 BN一个不带,比较它们训练中在 test上的表现(cost),例子:曲线拟合
加BN的语句: tf.layers.batch_normalization(x , momentum=0.4 , training=tf_is_train)
'''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

# 超参数
N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03
N_HIDDEN = 8                                # 一共8层隐藏层,超级过拟合
ACTIVATION = tf.nn.tanh                     # lost下降的比较平滑
# ACTIVATION = tf.nn.relu  # lost下降曲折,还有震荡
B_INIT = tf.constant_initializer(-0.2)      # 用一个很烂的 bias

# training data
x = np.linspace(-7 , 10 , N_SAMPLES)[ : ,np.newaxis]
np.random.shuffle(x)                        # 消除相关性
noise = np.random.normal(0 , 2 , x.shape)   # np.random.normal : 正态分布
y = np.square(x) - 5 + noise
train_data = np.hstack((x , y))             # shape : (2000 , 2)

# test data
test_x = np.linspace(-7 , 10 , 200)[ : ,np.newaxis]
noise = np.random.normal(0 , 2 , test_x.shape)
test_y = np.square(test_x) - 5 + noise

# # 展示生成的数据(散点图)
# plt.scatter(x, y, c='#FF9359', s=50, alpha=0.5, label='train')
# plt.legend(loc='upper left')
# plt.show()

# 设置传入值
tf_x = tf.placeholder(tf.float32 , [None , 1])
tf_y = tf.placeholder(tf.float32 , [None , 1])
tf_is_train = tf.placeholder(tf.bool , None)  # 区分是在训练还是测试

class NN(object):
    def __init__(self , batch_normalization = False):
        self.is_bn = batch_normalization

        # 初始化 W
        self.w_init = tf.random_normal_initializer(0. , .1)
        self.pre_activation = [tf_x]    # 列表记录每层激活层之前的输入数据分布
        if self.is_bn:                  # 如果是有BN的网络,那输入也要normalize
            self.layer_input = [tf.layers.batch_normalization(tf_x , training=tf_is_train)]
            # 列表记录每层输入的数据分布
        else:
            self.layer_input = [tf_x]
        for i in range(N_HIDDEN):       # 天才写法,记录每一层的输入数据分布
            # 全连接层往上叠,当然如果self.is_bn == True的话,会每层之间自动夹一层 BN
            self.layer_input.append(self.add_layer(self.layer_input[-1] , 10 , ac = ACTIVATION))

        # 隐藏层完了,定义输出层
        self.out = tf.layers.dense(self.layer_input[-1] , 1 , kernel_initializer=self.w_init , bias_initializer=B_INIT)
        self.loss = tf.losses.mean_squared_error(tf_y , self.out)

        # !! IMPORTANT !! the moving_mean and moving_variance need to be updated,
        # pass the update_ops with control_dependencies to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train = tf.train.AdamOptimizer(LR).minimize(self.loss)

    def add_layer(self , x , out_size , ac = None):
        # 定义叠一层神经网络的方法,需要输入数据,输出神经元数量,和激活函数类型
        x = tf.layers.dense(x , out_size , kernel_initializer=self.w_init , bias_initializer=B_INIT)
        self.pre_activation.append(x)
        # momentum这个参数很关键,默认的 0.99太大了
        if self.is_bn : x = tf.layers.batch_normalization(x , momentum=0.4 , training=tf_is_train)
        out = x if ac == None else ac(x)
        return out

# 创建两个网络对比
nets = [NN(batch_normalization=False) , NN(batch_normalization=True)]

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# plot layer input distribution
f, axs = plt.subplots(4, N_HIDDEN+1, figsize=(10, 5))
plt.ion()   # something about plotting

def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax,  ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0: p_range = (-7, 10); the_range = (-7, 10)
        else: p_range = (-4, 4); the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5)
        ax_pa_bn.hist(pre_ac_bn[i].ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].ravel(), bins=10, range=the_range, color='#FF9359')
        ax_bn.hist(l_in_bn[i].ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]:
            a.set_yticks(()); a.set_xticks(())
        ax_pa_bn.set_xticks(p_range); ax_bn.set_xticks(the_range); axs[2, 0].set_ylabel('Act'); axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)


losses = [[] , []]
for epoch in range(EPOCH):
    print('Epoch : ' , epoch)
    np.random.shuffle(train_data)  #  斩断相关性
    step = 0
    in_epoch = True
    while in_epoch:
        # batch index , b_s为起点 , b_f为终点
        b_s , b_f = (step * BATCH_SIZE) % len(train_data) , ((step+1) * BATCH_SIZE) % len(train_data)
        step += 1
        if b_f < b_s: # 整个 train_data都被抽了一遍,把 b_f令成结尾
            b_f = len(train_data)
            in_epoch = False # 这一代迭代结束了
        b_x, b_y = train_data[b_s: b_f, 0:1], train_data[b_s: b_f, 1:2] # 怎么有这种憨憨写法...
        #b_x , b_y = train_data[b_s : b_f , 0] , train_data[b_s : b_f , 1] # 这才是憨憨写法啊我日,这个shape会变成(64 , )
        sess.run([nets[0].train , nets[1].train] , {tf_x : b_x , tf_y : b_y , tf_is_train : True})

        if step == 1: # 每一个EPOCH展示一下在 test集上的表现
            l0 , l1 , l_in , l_in_bn , pa , pa_bn = sess.run(
                [nets[0].loss , nets[1].loss , nets[0].layer_input , nets[1].layer_input ,
                 nets[0].pre_activation , nets[1].pre_activation] ,
                {tf_x : test_x , tf_y : test_y , tf_is_train : False}  # 测试集不能再加 normalization 了
            )
            [loss.append(l) for loss , l in zip(losses , [l0 , l1])]
            plot_histogram(l_in, l_in_bn, pa, pa_bn)     # plot histogram


plt.ioff()

# plot test loss
plt.figure(2)
plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
plt.ylabel('test loss'); plt.ylim((0, 2000)); plt.legend(loc='best')

# plot prediction line
pred, pred_bn = sess.run([nets[0].out, nets[1].out], {tf_x: test_x, tf_is_train: False})
plt.figure(3)
plt.plot(test_x, pred, c='#FF9359', lw=4, label='Original')
plt.plot(test_x, pred_bn, c='#74BCFF', lw=4, label='Batch Normalization')
plt.scatter(x[:200], y[:200], c='r', s=50, alpha=0.2, label='train')
plt.legend(loc='best'); plt.show()





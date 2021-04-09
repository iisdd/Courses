# 练习一下PPO立杆子,用pytorch重写的,但是train不动...
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

# 超参数
A_LR = 0.0001
C_LR = 0.0002
A_UPDATE_STEPS = 10 # 每次sample完一个minibatch更新多少次actor
C_UPDATE_STEPS = 10 # 每次sample完一个minibatch更新多少次critic
S_DIM, A_DIM = 3, 1 # S: cos(theta), sin(theta), theta_dot角速度
EPSILON = 0.2       # clip的范围

class PPO(object):  # 继承object有更多魔法方法
    def __init__(self):
        self.sess = tf.Session()

        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic网络,吃state(算V(s)),discounted_r(主程序里操作), 吐advantage,advantage平方变成closs
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1) # 算V(s)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage)) # critic的loss: advantage的平方
            # critic要最小化advantage,要让估计的V更接近discounted_r
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)                  # 实时更新
        # pi 和 oldpi是两个distribution, 两个网络的参数拿出来是为了赋值更新
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)        # 老pi不更新fix住直接赋值
        with tf.variable_scope('sample_action'):                                # 选动作,带噪声的
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)                   # 这边改成了老pi作sample,测试一下performance,但是改成老pi环境会产生nan,很烦
            # pi是个distribution, pi.sample(1)是个tensor, 要用sess.run()才会变成numpy
        with tf.variable_scope('update_oldpi'):                                 # 老pi赋值
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')          # 选出来的带噪声的动作
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):                                # 先算一个没clip的loss
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa + 1e-5))
                surr = ratio * self.tfadv
            self.aloss = -(tf.reduce_mean(tf.minimum(
                surr, tf.clip_by_value(ratio, 1 - EPSILON, 1 + EPSILON)*self.tfadv
            )))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        self.sess.run(tf.global_variables_initializer())                                # 初始化启动
        self.saver = tf.train.Saver()                                                   # 这句话要放最后


    def _build_anet(self, name, trainable):                                             # 创建actor网络
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            # 2 * 是动作上下限 a_bound
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)     # softplus是平滑版的relu
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)                    # 增加exploration
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def update(self, s, a, r):                                                          # 送进来的一竖条32个minibatch, r是discounted的reward
        self.sess.run(self.update_oldpi_op)                                             # 老的pi总是落后于pi
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})

        # 更新网络参数
        [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv})for _ in range(A_UPDATE_STEPS)]
        # s用来算pi的distribution, a用来算ratio, adv用来算aloss
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r})for _ in range(C_UPDATE_STEPS)]
        # s用来算V(s),和tfdc_r作差算closs

    def choose_action(self, s):
        s = s[np.newaxis, : ]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        # 经过tf.Session.run()的tensor输出都会是numpy
        return np.clip(a, -2, 2)                # 限幅,立杆子的力(-2, 2)

    def get_v(self, s):                         # 吃状态s吐V(s)
        if s.ndim < 2: s = s[np.newaxis, : ]    # 防止长度为1的s掉成0维
        return self.sess.run(self.v, {self.tfs: s})[0, 0]

    def save_model(self, path):
        self.saver.save(self.sess, save_path = path)
        print('模型保存成功!')

    def load_model(self, path):
        self.saver.restore(self.sess, save_path = path)
        print('模型加载成功!')


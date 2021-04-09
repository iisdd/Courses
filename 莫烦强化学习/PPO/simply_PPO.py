"""
    立杆子,真能train起来, 我服了
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10  # 每次pi采样之后actor更新多少次
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 3, 1  # S: cos(theta), sin(theta), theta_dot角速度

# 选带有KL penality的还是Clipped的
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]                                                # choose the method for optimization


class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic,吃state,discounted_r  吐advantage,advantage平方变成closs
        with tf.variable_scope('critic'):           # 一层就train起来了...
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)         # 算state的value
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r') # 当前状态之后的reward的discount的和, tfdc_r: tensorflow discounted reward
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)  # 最小化advantage

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)                  # 这里的pi和oldpi都是distribution
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)        # fix住不训练
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)                   # choosing action, pi是个distribution, 这里应该用老pi才对
            # pi.sample(1)还是个tensor,要用sess.run()才会变numpy
        with tf.variable_scope('update_oldpi'):                                 # 老pi赋值
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):                                # 原始loss
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / (oldpi.prob(self.tfa) + 1e-5)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:                                                               # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)  # 程序里只能最小化loss

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):                                                  # 这送进来的都是32个一条的minibatch, r是discounted的reward
        self.sess.run(self.update_oldpi_op)                                     # 老pi总是落后于pi
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})      # 先用critic算adv, 再丢进actor算J(actor)
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor METHOD['kl_target']: 0.01
        # 不同的方法只体现在J(actor)上的不同
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:      # this in in google's paper, 差太远了打断更新
                    break
            if kl < METHOD['kl_target'] / 1.5:      # adaptive lambda, this is in OpenAI's paper, 更新幅度太小, 减小惩罚力度
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:    # 更新幅度太大, 加大惩罚力度
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:                                                   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):                                             # 创建actor网络
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)        # 我服了,我终于看懂前面那个 2 * 是啥了,nmd是环境的a_bound: (-2, 2)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)     # softplus就是平滑化的relu
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)                    # 用normal distribution 增加exploration
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params                                                        # 返回动作值和所有actor网络参数,参数用来更新老网络的

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]                             # 经过tf.Session.run()的tensor输出都会是numpy
        return np.clip(a, -2, 2)                                                        # 限幅,立杆子的力(-2, 2)

    def get_v(self, s):                                         # 吃状态s吐V(s)
        if s.ndim < 2: s = s[np.newaxis, :]                     # 防止长度为1的s掉成0维
        return self.sess.run(self.v, {self.tfs: s})[0, 0]       # 输出的维度应该是(1, 1)

env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = []  # 滑动平均每个ep的reward,让最终的奖励曲线平滑一点

for ep in range(EP_MAX):        # 1000eps
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN):     # in one episode, 200steps
        # 用老的pi跑32个transition再更新pi
        if ep > EP_MAX - 10:env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_a.append(a)
        buffer_r.append((r+8)/8)    # normalize reward, find to be useful
        # r = -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2) , r∈(-16, 0), 所以把它normalize了一下
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN-1:             # BATCH: 32
            v_s_ = ppo.get_v(s_)
            discounted_r = []                               # 记这32个transitions的dc_r
            for r in buffer_r[::-1]:
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse()                          # 倒着添加的,翻转一下

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br)                          # 这个update就已经包括actor,critic各更新了10次了
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)      # 又开始滑动平均了
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()
print('最后100个eps的平均reward为: ', np.mean(all_ep_r[-100:])) # -303.92

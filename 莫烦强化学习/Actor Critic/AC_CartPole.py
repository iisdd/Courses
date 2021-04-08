"""
例子:立杆子
Actor-Critic 结合了 Policy Gradient (Actor) 和 Function Approximation (Critic) 的方法
Actor Critic 方法的优势: 可以进行单步更新, 比传统的 Policy Gradient(回合更新) 要更快收敛.
劣势: 取决于Critic的价值判断,但是Critic难收敛,由于是连续状态下的更新,相关性太强了,
为了解决收敛问题,Google Deepmind提出了DDPG(可以处理连续动作)
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import gym

np.random.seed(2)
tf.set_random_seed(2)  # reproducible

# 超参数
OUTPUT_GRAPH = False
MAX_EPISODE = 3000
DISPLAY_REWARD_THRESHOLD = 300  # 渲染环境的起步条件
MAX_EP_STEPS = 1000
RENDER = False
GAMMA = 0.9
LR_A = 0.001
LR_C = 0.01

env = gym.make('CartPole-v0')
env.seed(1)
env = env.unwrapped

N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class Actor(object):  # 和之前的 PG大同小异,只能处理离散动作
    def __init__(self , sess , n_features , n_actions , lr = 0.001):
        self.sess = sess

        self.s = tf.placeholder(tf.float32 , [1 , n_features] , 'state')
        self.a = tf.placeholder(tf.int32 , None , 'act')
        self.td_error = tf.placeholder(tf.float32 , None , 'td_error')

        with tf.variable_scope('Actor'):
            # 输入层
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0. , .1),
                bias_initializer=tf.constant_initializer(0.1),
                name = 'l1',
            )
            # 输出层
            self.acts_prob = tf.layers.dense(
                inputs = l1,
                units = n_actions,
                activation=tf.nn.softmax,
                kernel_initializer=tf.random_normal_initializer(0. , .1),
                bias_initializer=tf.constant_initializer(0.1),
                name = 'acts_prob',
            )

        # theta = theta + alpha*log(p(s,a))*v
        # 这里选的 v是 td_error
        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0 , self.a])
            self.exp_v = tf.reduce_mean(log_prob * self.td_error) # reduce_mean求某一维度的平均值

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v) # 也就是最大化self.exp_v

    def learn(self , s , a , td):
        s = s[np.newaxis , : ]
        feed_dict = {self.s:s , self.a:a , self.td_error:td}
        _,exp_v = self.sess.run([self.train_op , self.exp_v] , feed_dict)
        return exp_v

    def choose_action(self , s):
        s = s[np.newaxis , : ]
        probs = self.sess.run(self.acts_prob , {self.s : s})
        return np.random.choice(np.arange(probs.shape[1]) , p = probs.ravel())  # 返回一个整数


class Critic(object):  # 输入s预测V(s)
    def __init__(self , sess , n_features , lr = 0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32 , [1 , n_features] , 'state')
        self.v_ = tf.placeholder(tf.float32 , [1 , 1] , 'v_next')
        self.r = tf.placeholder(tf.float32 , None , 'r')

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs = self.s,
                units = 20,
                activation=tf.nn.relu,
                # 一方面要是线性的激活层才能让 actor收敛
                # 另一方面线性的估计器很难学到准确的 Q
                kernel_initializer=tf.random_normal_initializer(0. , .1),
                bias_initializer=tf.constant_initializer(0.1),
                name = 'l1'
            )

            self.v= tf.layers.dense(   # 输出价值函数的估计值
                inputs = l1,
                units = 1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0. , .1),
                bias_initializer=tf.constant_initializer(0.1),
                name = 'V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + GAMMA * self.v_ - self.v
            self.loss = tf.square(self.td_error)
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self , s , r , s_):
        s , s_ = s[np.newaxis , : ] , s_[np.newaxis , : ]

        v_ = self.sess.run(self.v , {self.s : s_})  # 把这里打成 s了,md改了我半天...
        td_error,_ = self.sess.run([self.td_error , self.train_op] , {self.s : s , self.v_ :v_ , self.r : r})
        return td_error

sess = tf.Session()

actor = Actor(sess , n_features=N_S , n_actions=N_A , lr = LR_A)
critic = Critic(sess , n_features=N_S , lr = LR_C)  # LR_C > LR_A 评论家肯定要学的比演员快吧

sess.run(tf.global_variables_initializer())

for i_episode in range(MAX_EPISODE):
    s = env.reset()
    t = 0
    track_r = []
    while 1:
        if RENDER : env.render()

        a = actor.choose_action(s)

        s_ , r , done , info = env.step(a)

        if done : r = -20

        track_r.append(r)

        td_error = critic.learn(s , r , s_)
        actor.learn(s , a , td_error)   # grad[logPi(s,a) * td_error]

        s = s_
        t += 1

        if done or t >= MAX_EP_STEPS:
            ep_rs_sum = sum(track_r)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.95 + ep_rs_sum * 0.05

            if running_reward > DISPLAY_REWARD_THRESHOLD : RENDER = True
            print('episode: ' , i_episode , ' reward: ' , int(running_reward))
            break

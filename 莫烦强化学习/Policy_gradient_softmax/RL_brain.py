"""
    策略梯度方法决策部分, 吃 n_features个状态, 吐 n_actions个动作选择的概率
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

np.random.seed(1) # 可复现
tf.set_random_seed(1)


class PolicyGradient:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate = 0.01,
            reward_decay = 0.95,
            output_graph = False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay

        self.ep_obs,self.ep_as,self.ep_rs = [], [], []

        self._build_net()

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

    def _build_net(self):
        with tf.name_scope('input'):
            # tf.placeholder(dtype, shape=None, name=None) , 返回tensor
            self.tf_obs = tf.placeholder(tf.float32 , [None , self.n_features] , name = 'observations')
            self.tf_acts = tf.placeholder(tf.int32 , [None , ] , name = 'actions_num')
            self.tf_vt = tf.placeholder(tf.float32 , [None , ] , name = 'actions_value')

        # 全连接层 1
        layer = tf.layers.dense(
            inputs = self.tf_obs,
            units = 10,
            activation = tf.nn.tanh,
            kernel_initializer = tf.random_normal_initializer(mean = 0 , stddev = 0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name = 'fc1'
        )
        # 全连接层 2
        all_act = tf.layers.dense(
            inputs = layer,
            units = self.n_actions, # 输出动作选择
            activation = None,
            kernel_initializer = tf.random_normal_initializer(mean = 0 , stddev = 0.3),
            bias_initializer = tf.constant_initializer(0.1),
            name = 'fc2'
        )
        self.all_act_prob = tf.nn.softmax(all_act , name = 'act_prob') # 动作选择概率为softmax

        with tf.name_scope('loss'):
            # 为了让总收益(log_p * R)最大化,相当于最小化-(log_p * R),ps: tf只有最小化的功能,参考最优化方法
            neg_log_prob = tf.reduce_sum(-tf.log(self.all_act_prob) * tf.one_hot(self.tf_acts, self.n_actions) , axis = 1)
            loss = tf.reduce_mean(neg_log_prob * self.tf_vt) # 收益决定loss

        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def choose_action(self, observation):
        # 输入状态特征,输出每个动作选择的概率
        prob_weights = self.sess.run(self.all_act_prob , feed_dict={self.tf_obs : observation[np.newaxis , : ]}) # 加一个维度送入神经网络
        # 根据概率确定动作
        action = np.random.choice(range(prob_weights.shape[1]) , p = prob_weights.ravel())
        return action

    def store_transition(self , s , a , r): # 不用存s_
        self.ep_obs.append(s)
        self.ep_as.append(a)
        self.ep_rs.append(r)

    def learn(self):
        # discount and normalize episode reward
        discounted_ep_rs_norm = self._discount_and_norm_rewards()
        # 训练神经网络参数
        self.sess.run(self.train_op , feed_dict={
            self.tf_obs : np.vstack(self.ep_obs),               # shape = [None , n_obs]
            self.tf_acts : np.array(self.ep_as),                # shape = [None , ]
            self.tf_vt : discounted_ep_rs_norm,                 # shape = [None , ]
        })

        self.ep_obs , self.ep_as , self.ep_rs = [], [], []      # PG是每回合学习一次,所以要把 s,a,r重置
        return discounted_ep_rs_norm

    def _discount_and_norm_rewards(self):
        # 对每个episode的rewards归一化
        discounted_ep_rs = np.zeros_like(self.ep_rs)            # zeros_like:创建一个像矩阵的array
        running_add = 0
        for t in reversed(range(0 , len(self.ep_rs))):          # reversed :倒着来
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_ep_rs[t] = running_add

        # 归一化处理
        discounted_ep_rs -= np.mean(discounted_ep_rs)           # 平均值变成 0
        discounted_ep_rs /= np.std(discounted_ep_rs)            # std:标准差 , 相当于把方差变成 1
        return discounted_ep_rs                                 # 完成任务越靠后的动作越有价值吧



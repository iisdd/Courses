"""
DQN的tensorflow实现
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.
Tensorflow版本: 1.15
"""

import numpy as np
import pandas as pd
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


np.random.seed(1)
tf.set_random_seed(1)


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learn_step_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2)) # 行数self.memory_size , 列数n_features * 2 + 2

        # consist of [target_net, evaluate_net]
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]
        # 替换旧的网络参数w

        self.sess = tf.Session()

        if output_graph:
            # $ tensorboard --logdir=logs
            # tf.train.SummaryWriter soon be deprecated, use following
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []

    def _build_net(self):
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss
        with tf.variable_scope('eval_net'):
            # c_names(collections_names) are the collections to store variables
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 10, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_eval = tf.matmul(l1, w2) + b2

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)     # 没有用Adam

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            # c_names(collections_names) are the collections to store variables
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            # first layer. collections is used later when assign to target net
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(self.s_, w1) + b1)

            # second layer. collections is used later when assign to target net
            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                self.q_next = tf.matmul(l1, w2) + b2

    def store_transition(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):
            self.memory_counter = 0

        transition = np.hstack((s,[a,r],s_)) # 横向压缩状态转移
        # 更新记忆库
        index = self.memory_counter % self.memory_size
        self.memory[index , : ] = transition
        # 每次更新一行
        self.memory_counter += 1

    def choose_action(self , observation):
        # 加一个维度变成矩阵才能输入tensorflow
        observation = observation[np.newaxis , : ]

        if np.random.uniform() < self.epsilon: # uniform默认在0到1之间随机取值
            # 贪心,向前探索一步,算出每个action的q
            actions_value = self.sess.run(self.q_eval , feed_dict={self.s : observation})
            # 备注:self.sess.run(self.out , feed_dict) feed_dict喂入网络,self.out输出
            # 等于这里是输入s,输出一行q(s,a),a∈A
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0 , self.n_actions) # 前闭后开
        return action

    def learn(self):
        # 检查是否需要替换掉原来老的w(用于q_target计算),新的w用于计算q_predict
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op) # op=old parameters
            print('target_parameters_replaced')

        ################ 随机抽样一批记忆库记忆 #############
        if self.memory_counter > self.memory_size: # 记忆库满了
            # 等于是0-499抽样32个数
            sample_index = np.random.choice(self.memory_size , size = self.batch_size)
        else: # 记忆库还没装满,有多少抽多少，由于是200步开始学习,记忆库有500个数据,所以是存在这种可能的
            sample_index = np.random.choice(self.memory_counter , size = self.batch_size)
        batch_memory = self.memory[sample_index , : ]

        q_next,q_eval = self.sess.run(
            [self.q_next , self.q_eval],
            feed_dict={
                self.s_ : batch_memory[ : , -self.n_features : ],   # 取了s_的特征值送入老w的神经网络,算q_next,再算q_target = r + max(q_next)
                self.s : batch_memory[ : ,  : self.n_features],     # 取了s的特征值送入新w的神经网络,算q_predict
            }
        )
        # 输出的q_next,q_eval都是32行n_actions列
        q_target = q_eval.copy() # 32行(batch_size),n_actions列,只改变被选动作的q
        batch_index = np.arange(self.batch_size , dtype = np.int32)
        eval_act_index = batch_memory[ : , self.n_features].astype(int)  # 抽出动作来
        reward = batch_memory[ : , self.n_features + 1]
        # 瘦高条子的矩阵运算(reward , np.max(q_next , axis = 1))
        q_target[batch_index , eval_act_index] = reward + self.gamma * np.max(q_next , axis = 1) # 行最大值
        '''
        举个例子：
        如果我一批抽样2个例子3个动作
        q_eval = 
        [[1,2,3],
         [4,5,6]]
        先copy过来:q_target = q_eval = 
        [[1,2,3],
         [4,5,6]]
        我只改变他抽样中执行的动作对应的q,例如:
        sample 0 中执行动作0,max q(s_) = -1 (为简化例子,reward为0,gamma=1)
        sample 1 中执行动作2,max q(s_) = -2
        那么q_target就变成了:
        [[-1,2,3],
         [4,5,-2]]
        delta = q_target - q_predict = 
        [[-2,0,0],
         [0,0,-8]]
        delta传回去之后只改变选择的动作对应的q
        让人不禁感叹,这就是矩阵运算啊!!!!!!
        '''
        # 训练eval,BP网络
        _,self.cost = self.sess.run(
            [self._train_op , self.loss],
            feed_dict={self.s : batch_memory[ : , : self.n_features],
                       self.q_target : q_target}
        )
        # 输入s,q_target, s ==> q_eval , delta = q_target - q_predict 反向传播改变神经网络w
        self.cost_his.append(self.cost)
        # 增大epsilon,让策略收敛
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    # 画图部分
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)) , self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


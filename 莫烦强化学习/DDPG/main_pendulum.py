"""
    例子：立杆子,没有小车.经典的连续动作选择
    DDPG结合了Actor-critic 和 DQN,不输出动作的概率而是输出确定动作,用 DQN来逼近价值函数
"""

import tensorflow as tf
import numpy as np
import gym
import time
from DDPG import DDPG
##############################超参数##############################
MAX_EPISODES = 1000
MAX_EP_STEPS = 200
MEMORY_CAPACITY = 10000
RENDER = False
ENV_NAME = 'Pendulum-v0'
###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high
print('动作上限: ', a_bound)

ddpg = DDPG(a_dim, s_dim, a_bound)

var = 3  # control exploration
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    for j in range(MAX_EP_STEPS):
        if RENDER:
            env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration

        # print('状态: ', s)
        # print('动作: ', a)
        s_, r, done, info = env.step(a)

        ddpg.store_transition(s, a, r / 10, s_)

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # 方差大小随训练衰减
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'Explore: %.2f' % var, )
            break

print('Running time: ', time.time() - t1)
# 模型保存
ddpg.save_model('./model_saved/pendulum_model')

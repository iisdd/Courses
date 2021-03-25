"""
Deep Q network,
Using:
Tensorflow: 1.0
gym: 0.7.3
"""
# _ 是忽略的意思,如果一个输出用_代替,就代表这个值不关键

import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

env = gym.make('CartPole-v0')
env = env.unwrapped

MEMORY_SIZE = 3000
ACTION_SPACE = 2

print(env.action_space) # discrete(2) 两个离散动作:向左or向右
print(env.observation_space) # Box(4,)
print(env.observation_space.high) # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low) # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]



sess = tf.Session()
RL = DoubleDQN(
    n_actions=ACTION_SPACE, n_features=4, memory_size=MEMORY_SIZE,
    e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)
sess.run(tf.global_variables_initializer())

total_steps = 0
total_reward = []

for i_episode in range(300):
    observation = env.reset()
    ep_r = 0
    while True:
        # if i_episode > 90:env.render()

        action = RL.choose_action(observation)

        observation_ , reward , done , info = env.step(action)

        # 设置reward
        x , x_dot , theta , theta_dot = observation_ # 状态由四个特征定义
        r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        reward = r1 + r2

        RL.store_transition(observation , action , reward ,observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done:
            total_reward.append(ep_r)
            print(
                'episode: ',i_episode,
                'e_r: ',round(ep_r , 2), # round :四舍五入小数点后两位
                'epsilon: ',round(RL.epsilon , 2),
            )
            break

        observation = observation_
        total_steps += 1

print('平均reward: ' , np.mean(total_reward[-100: ]))  # 100eps: 51.52 , 300eps(后100eps): 402.59
plt.plot(total_reward)
plt.show()







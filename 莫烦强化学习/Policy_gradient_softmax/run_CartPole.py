"""
    策略梯度方法例子：立杆子,效果比离散动作要好很多
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = 800      # reward不到800就别出来丢人现眼吧
RENDER = False                      # 渲染环境浪费时间

env = gym.make('CartPole-v0')
env.seed(1)                         # PolicyGradient的方差很大,比较难复现
env = env.unwrapped

'''
print(env.action_space) # Discrete(2),左右
print(env.observation_space) # Box(4, )
print(env.observation_space.high) # [4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low)  # [-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]
'''

RL = PolicyGradient(
    n_actions = env.action_space.n,
    n_features = env.observation_space.shape[0],
    learning_rate = 0.02,
    reward_decay = 0.99,
)

for i_episode in range(3000):

    observation = env.reset()

    while 1:
        if RENDER : env.render()

        action = RL.choose_action(observation)

        observation_ , reward , done , info = env.step(action)

        RL.store_transition(observation, action , reward)

        if done:
            ep_rs_sum = sum(RL.ep_rs)

            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:   # 滑动变化
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD : RENDER = True    # 得分到400再显示
            print('episode: ' , i_episode , ' reward: ' , int(running_reward))

            vt = RL.learn()                                                 # 进行了打折和归一化处理

            if i_episode % 500 == 0:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break
        observation = observation_










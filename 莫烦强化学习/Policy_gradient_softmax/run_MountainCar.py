"""
    策略梯度方法例子：小车爬坡, PG train不动
"""

import gym
from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -500 # 训练到一定程度再出动画,加速训练

RENDER = False

env = gym.make('MountainCar-v0')
env.seed(1)
env = env.unwrapped
'''
print(env.action_space) # Discrete(3) 为啥有3个动作...
print(env.observation_space) # Box(2,)
print(env.observation_space.high) # [0.6  0.07]
print(env.observation_space.low)  # [-1.2  -0.07]
'''

RL = PolicyGradient(
    n_actions = env.action_space.n,
    n_features = env.observation_space.shape[0],
    learning_rate = 0.02,
    reward_decay = 0.995,
)

for i_episode in range(3000):

    observation = env.reset()

    while 1 :
        if RENDER : env.render()

        action = RL.choose_action(observation)

        observation_ , reward , done , info = env.step(action)  # reward每一步都是-1

        RL.store_transition(observation , action , reward)

        if done:
            # 计算running reward
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True
            print('episode: ', i_episode , ' reward:' , int(running_reward))

            vt = RL.learn()

            if i_episode == 30:
                plt.plot(vt)
                plt.xlabel('episode steps')
                plt.ylabel('normalized state-action value')
                plt.show()
            break
        observation = observation_



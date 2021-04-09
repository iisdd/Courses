# 对应莫烦的PPO的主程序
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from PPO_ex1 import PPO
# 超参数
EP_MAX = 10
EP_LEN = 200 # 每个ep的max_step


# 主程序
env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
ppo.load_model('./model_saved/pendulum_model')

for ep in range(EP_MAX): # 1000eps
    s = env.reset()
    ep_r = 0
    for t in range(EP_LEN): # 200steps
        env.render()
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        s = s_
        ep_r += r
    print(
        'EP: %i' % ep, # i: int
        '|EP_r: %s' % str(ep_r),
    )




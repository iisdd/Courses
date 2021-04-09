# 对应莫烦的PPO的主程序,用于训练模型
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym
from PPO_ex1 import PPO
# 超参数
GAMMA = 0.9
EP_MAX = 1000
EP_LEN = 200 # 每个ep的max_step
BATCH = 32

# 主程序
env = gym.make('Pendulum-v0').unwrapped
ppo = PPO()
all_ep_r = [] # 滑动平均每个ep的reward,让最后的奖励曲线平滑一些

for ep in range(EP_MAX): # 1000eps
    s = env.reset()
    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0
    for t in range(EP_LEN): # 200steps
        # 用老的pi跑出32个transitions再更新pi
        # if ep > EP_MAX - 10: env.render()                     # 出动画
        a = ppo.choose_action(s)
        s_, r, done, _ = env.step(a)
        buffer_s.append(s)
        buffer_r.append((r+8)/8)
        # r = -(theta^2 + 0.1*theta_dt^2 + 0.001*action^2) , r∈(-16, 0), 所以把它normalize了一下
        buffer_a.append(a)
        s = s_
        ep_r += r

        # update ppo
        if (t+1) % BATCH == 0 or t == EP_LEN - 1: # 每sample32个transitions就重新sample
            v_s_ = ppo.get_v(s_)
            discounted_r = [] # 记录32个transitions的dc_r
            for r in buffer_r[: : -1]: # 倒着来
                v_s_ = r + GAMMA * v_s_
                discounted_r.append(v_s_)
            discounted_r.reverse() # 从尾巴添加到头的,翻转一下

            bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
            buffer_s, buffer_a, buffer_r = [], [], []
            ppo.update(bs, ba, br) # 这里包括了actor和critic各更新10次
    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1] * 0.9 + ep_r * 0.1) # 滑动起来
    print(
        'EP: %i' % ep, # i: int
        '|EP_r: %s' % str(ep_r),
    )
# 模型保存
ppo.save_model('./model_saved/pendulum_model')


# 训练完画图
plt.plot(all_ep_r)
plt.xlabel('Episode')
plt.ylabel('Moving averaged episode reward')
plt.show()
print('最后100个eps的平均reward为: ', np.mean(all_ep_r[-100:]))



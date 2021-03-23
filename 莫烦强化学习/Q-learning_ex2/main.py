"""
将Q-learning用在一个复杂一点的环境(寻找迷宫最优路线)
主程序
Reinforcement learning maze example.
Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].
This script is the main part which controls the update method of this example.
The RL is in RL_brain.py.
"""
from maze_env import Maze
from RL_brain import QLearningTable

def update():
    for episode in range(100):
        # 初始化状态
        observation = env.reset()
        while 1:
            # 更新环境
            env.render()
            # 根据状态选择动作
            action = RL.choose_action(str(observation))
            # 输入动作,返回下一状态,即时收益,done标志走到宝藏或者黑洞
            observation_,reward,done = env.step(action)
            # 学习更新Q表
            RL.learn(str(observation),action,reward,str(observation_))
            # 更新状态
            observation = observation_
            if done:
                break
    print('游戏结束')
    env.destroy()
if __name__ == '__main__':
    env = Maze()
    RL = QLearningTable(actions = list(range(env.n_actions)))
    env.after(100,update)
    env.mainloop()


"""
用Sarsa算法在迷宫环境进行测试
主程序
Sarsa is a online updating method for Reinforcement learning.
Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.
You will see the sarsa is more coward(胆小) when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""
from maze_env import Maze
from RL_brain import SarsaTable

def update():
    for episode in range(100):
        # 初始化状态
        observation = env.reset()
        # 选择动作
        action = RL.choose_action(str(observation))

        while 1:
            # 渲染环境
            env.render()
            # 执行动作,反馈s_,reward
            observation_,reward,done = env.step(action)
            # 选择下一个动作action_
            action_ = RL.choose_action(str(observation))
            # 从(s,a,r,s_,a_)中学习 ==> Sarsa
            RL.learn(str(observation),action,reward,str(observation_),action_)
            # 更新状态和动作
            observation = observation_
            action = action_
            if done:
                break
    print('游戏结束')
    env.destroy()
if __name__ == '__main__':
    env = Maze()
    RL = SarsaTable(actions = list(range(env.n_actions)))

    env.after(100,update)
    env.mainloop()




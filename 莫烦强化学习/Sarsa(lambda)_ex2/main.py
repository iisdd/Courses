"""
主程序
Sarsa is a online updating method for Reinforcement learning.
Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory.
You will see the sarsa is more coward when punishment is close because it cares about all behaviours,
while q learning is more brave because it only cares about maximum behaviour.
"""
from maze_env import Maze
from RL_brain import SarsaLambdaTable

def update():
    for episode in range(100):
        # 初始化状态
        observation = env.reset()
        # 选初始动作
        action = RL.choose_action(str(observation))
        # 初始化资格迹
        RL.eligibility_trace *= 0

        while 1:
            # 渲染环境
            env.render()
            # 执行动作获得反馈:s_,reward
            observation_,reward,done = env.step(action)
            # 根据s_选择a_
            action_ = RL.choose_action(str(observation_))
            # Sarsa学习==>(s,a,r,s_,a_)
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
    RL = SarsaLambdaTable(actions = list(range(env.n_actions)))
    env.after(100,update)
    env.mainloop()

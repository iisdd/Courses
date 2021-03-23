"""
sarsa的特点,q_predict是用s_对应的a_算的,而不是s_中使Q最大的a_
agent决策
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.

"""
import numpy as np
import pandas as pd

class RL(object):
    def __init__(self,action_space,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        self.actions = action_space # 主程序最后定义了action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns = self.actions , dtype = np.float64)
    def check_state_exist(self,state):
        if state not in self.q_table.index:
            # 把没见过的行加进来
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.actions,
                    name = state,
                )
            )
            # 改了一下,原来是index = self.q_table.columns
    def choose_action(self,observation):
        self.check_state_exist(observation)
        # 选动作
        if np.random.rand() < self.epsilon:
            # 贪心
            state_action = self.q_table.loc[observation ,  : ]
            # 随机选择相同的最大值
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # 随机
            action = np.random.choice(self.actions)
        return action
    def learn(self,*args):
        # 不同的方法继承到不同的子类
        pass
# off-policy,Q-learning
class QLearningTable(RL): # 继承父类RL
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        super(QLearningTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)
    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, : ].max()
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)

# on-policy,SARSA
class SarsaTable(RL):
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        super(SarsaTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_,a_]  # 确定的a_
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)




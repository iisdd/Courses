"""
决策更新
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
"""
import numpy as np
import pandas as pd

class QLearningTable:
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions,dtype=np.float64)      # 一开始Q表是个空表
    def choose_action(self,observation):
        self.check_state_exist(observation)                                     # 判断当前状态经历过没,没经历过就在df新加一行
        # 选择动作
        if np.random.uniform() < self.epsilon:
            # 贪心选择
            state_action = self.q_table.loc[observation, : ]
            # 相同的Q随机选择,如果不写这一句每次都会选靠前的
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # 随机选择
            action = np.random.choice(self.actions)
        return action
    def learn(self,s,a,r,s_):
        self.check_state_exist(s_)                                              # 判断次态是否存在
        q_predict = self.q_table.loc[s,a]                                       # .loc按行列名字取元素
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, : ].max()
        else:
            q_target = r
        self.q_table.loc[s,a] += self.lr * (q_target - q_predict)
    def check_state_exist(self,state):
        # .index : 所有行名
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.q_table.columns,
                    name = state,
                )
            )
            # 这里的index=是指列名,name=是新加的行名
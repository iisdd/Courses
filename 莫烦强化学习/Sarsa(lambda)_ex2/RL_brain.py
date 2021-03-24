"""
决策模块
This part of code is the Q learning brain, which is a brain of the agent.
All decisions are made in here.
"""
import numpy as np
import pandas as pd
class RL(object):
    def __init__(self,action_space,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9):
        self.actions = action_space
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

        self.q_table = pd.DataFrame(columns = self.actions,dtype = np.float64)

    def check_state_exist(self,state): # index是字符串
        if state not in self.q_table.index:
            self.q_table.append(
                pd.Series(
                    [0]*len(self.actions),
                    index = self.actions,
                    name = state,
                )
            )

    def choose_action(self,observation):
        self.check_state_exist(observation)
        if np.random.rand() < self.epsilon:
            # 贪心
            state_action = self.q_table.loc[observation,:]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # 随机
            action = np.random.choice(self.actions)
        return action

    def learn(self,*args):
        pass

# 后向资格迹
class SarsaLambdaTable(RL):
    def __init__(self,actions,learning_rate=0.01,reward_decay=0.9,e_greedy=0.9,trace_decay=0.9):
        super(SarsaLambdaTable,self).__init__(actions,learning_rate,reward_decay,e_greedy)
        self.lambda_ = trace_decay   # lambda后面加个_区别于lambda方法
        self.eligibility_trace = self.q_table.copy()

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            to_be_appended = pd.Series(
                    [0] * len(self.actions),
                    index = self.actions,
                    name = state,
                ) # q_table 和 eligibility_trace 都要添加新的一行
            # 这个append是个憨憨,必须加个 = ,不然就出keyerror.
            self.q_table = self.q_table.append(to_be_appended)
            self.eligibility_trace = self.eligibility_trace.append(to_be_appended)

    def learn(self,s,a,r,s_,a_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s,a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_,a_]
        else:
            q_target = r
        delta = q_target - q_predict
        # 不让可信度超过1
        self.eligibility_trace.loc[s, : ] *= 0
        self.eligibility_trace.loc[s , a] = 1
        # 人造特征向量X(s)
        self.q_table += self.lr * delta * self.eligibility_trace  # 见者有份,大家都更新,越近幅度越大,苟富贵勿相忘
        # 资格迹最后更新
        self.eligibility_trace *= self.gamma * self.lambda_

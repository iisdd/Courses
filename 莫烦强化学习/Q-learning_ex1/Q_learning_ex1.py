# 算法: Q-learning 环境: 寻宝,两个动作六个状态.
# 最后显示出每个状态下往左往右的Q-value
import numpy as np
import pandas as pd
import time
np.random.seed(2)       # 伪随机序列
# 用大写做变量防止和BIF冲突
N_STATES = 6            # 距离终点六步
ACTIONS = ['left' , 'right']
EPSILON = 0.9           # greedy policy
ALPHA = 0.1             # learning rate
LAMBDA = 0.9            # 衰减度
MAX_EPISODES = 13       # 只玩13回合,（训练13次）
FRESH_TIME = 0.3        # 每次走一步的时间


def build_q_table(n_states , actions):
    table = pd.DataFrame(
        np.zeros((n_states , len(actions))),
        columns=actions
    )
    # 初始化Q-table,6行2列的0矩阵
    return table

def choose_action(state , q_table):
    # 选择动作的函数
    state_actions = q_table.iloc[state , : ] #取对应状态那一行的Q
    if (np.random.uniform() > EPSILON) or (state_actions.all() == 0): #随机选择
        action_name = np.random.choice(ACTIONS)
        #print('随机')
    else: # act greedy
        action_name = ACTIONS[state_actions.argmax()] #由于不是用的ix,这里也要改成标签名
        #print('贪心')
    #print(action_name)
    return action_name

def get_env_feedback(S , A):
    # agent与环境交互的机制,S_表示S',次态
    if A == 'right':
        if S == N_STATES - 2: # 一共0,1,2,3,4,5 在第4位上
            S_ = 'terminal'
            R = 1 # 立即奖励
        else:
            S_ = S + 1
            R = 0
    else:
        R = 0
        if S == 0:
            S_ = S # 走到顶了
        else:
            S_ = S - 1
    return S_ , R

#环境直接不看了
def update_env(S , episode , step_counter):
    # 编写环境的更新
    env_list = ['-']*(N_STATES-1) + ['*'] # *表示宝藏位置
    if S == 'terminal':
        interaction = 'Episode %s: total_steps = %s' % (episode+1 , step_counter)
        print('\r{}'.format(interaction) )
        time.sleep(2)
        print('\r                                           ')
    else:
        env_list[S] = 'o'
        interaction = ''.join(env_list)
        print('\r{}'.format(interaction) )
        time.sleep(FRESH_TIME)

def rl():
    # 强化学习主程序
    q_table = build_q_table(N_STATES , ACTIONS)
    for episode in range(MAX_EPISODES):
        step_counter = 0
        S = 0
        is_terminated = False
        update_env(S , episode ,step_counter)
        while not is_terminated:
            A = choose_action(S , q_table)
            S_ , R = get_env_feedback(S , A)
            #print(S , A)
            q_predict = q_table.loc[S , A] #按行列取值
            if S_ != 'terminal':
                q_target = R + LAMBDA * q_table.loc[S_ , : ].max() #取某行最大值
            else:
                q_target = R #走到终点前一步了
                is_terminated = True #这个episode完结了

            q_table.loc[S , A] += ALPHA * (q_target - q_predict) # 更新q表
            S = S_
            step_counter += 1
            update_env(S ,episode , step_counter)
        print(q_table)
    return q_table

if __name__ == '__main__':
    q_table = rl()
    print('\r\nQ-table:\n')
    print(q_table)

#注意视频里的ix已经过期,用iloc和loc代替

"""
用torch写里面的神经网络,例子：凹谷开车
"""
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F # 激励函数
import gym

# 超参数
BATCH_SIZE = 32
LR = 0.001                          # learning rate
EPSILON = 0.9                       # 选择最优动作的概率
GAMMA = 0.9                         # reward的衰减系数
TARGET_REPLACE_ITER = 300           # 更新神经网络的迭代次数
MEMORY_CAPACITY = 3000              # 记忆库大小
env = gym.make('MountainCar-v0')    # 小车爬山
env = env.unwrapped
N_ACTIONS = env.action_space.n      # 能选的动作数
N_STATES = env.observation_space.shape[0]       # 环境特征数
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # 确保shape

class Net(nn.Module):
    def __init__(self, ):
        super(Net , self).__init__()
        self.fc1 = nn.Linear(N_STATES , 50)
        self.fc1.weight.data.normal_(0 , 0.1)   # 初始化权值
        self.out = nn.Linear(50 , N_ACTIONS)
        self.out.weight.data.normal_(0 , 0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value                    # 返回所有a的q(s,a)


class DQN(object):
    def __init__(self):
        self.eval_net , self.target_net = Net() , Net()
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY , N_STATES*2 + 2))                      # 规定行列
        self.optimizer = torch.optim.Adam(self.eval_net.parameters() , lr = LR)
        self.loss_func = nn.MSELoss()
        self.cost_his = []                                                              # 加个画图环节

    def choose_action(self, x):                                                         # 根据状态选动作
        x = torch.unsqueeze(torch.FloatTensor(x) , 0)                                   # unsqueeze扩充维度,0代表变成长条,1代表变成竖条
        if np.random.uniform() < EPSILON: # 贪心
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0  else action.reshape(ENV_A_SHAPE)    # 返回行最大值的index
        else:
            action = np.random.randint(0 , N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action


    def store_transition(self , s, a ,r ,s_):
        transition = np.hstack((s , [a,r] , s_)) # 压成一长条
        # 更新数据库
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index , : ] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('target参数更新')
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]

        # 分离一下抽样的 s,a,r,s_,转换成tensor格式,送进torch直接优化参数
        b_s = torch.FloatTensor(b_memory[ : , :N_STATES])
        b_a = torch.LongTensor(b_memory[ : ,N_STATES :N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[ : ,N_STATES+1 :N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[ : ,-N_STATES : ])

        # 根据s,a,s_ 算出q_eval和q_next,q_target
        q_eval = self.eval_net(b_s).gather(1 , b_a) # 瘦高条子,()取每一行第b_a列的q,gather()的类型必须是LongTensor
        q_next = self.target_net(b_s_).detach()    # .detach():获取Variable 内部tensor , 保持target_net不被梯度更新
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE , 1) # 取每行最大值加上去.后面的[0]代表取值，如果是[1]就是取index
        loss = self.loss_func(q_eval , q_target)
        self.cost_his.append(loss)

        # 计算，更新 eval_net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()

# 主程序部分
dqn = DQN()

print('填装记忆库中,尚未学习...')
for i_episode in range(20):
    s = env.reset()
    ep_r = 0
    while True:
        if i_episode > 17:
            env.render() # 渲染环境
        a = dqn.choose_action(s)

        # 环境对动作做出反馈
        s_ , r , done , info = env.step(a)

        # 修改一下reward的设置,让DQN更快收敛
        position , velocity = s_
        r = abs(position - (-0.5)) # 让它左右摆起来

        # 存入记忆库
        dqn.store_transition(s , a , r , s_)
        ep_r += r

        if dqn.memory_counter > MEMORY_CAPACITY: # 记忆库满了就开始学习
            dqn.learn()
            if done:
                print(
                    'Ep : ' , i_episode,
                    '|Ep_r : ' , ep_r,
                )

        if done:
            break

        s = s_
dqn.plot_cost()















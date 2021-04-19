'''
    DQN倒立摆实验
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym

# 超参数
BATCH_SIZE = 32
LR = 0.01
EPSILON = 0.9
GAMMA = 0.9
TARGET_REPLACE_ITER = 100
MEMORY_CAPACITY = 2000
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample() , int) else env.action_space.sample().shape

# 搭建神经网络部分
class Net(nn.Module):
    def __init__(self, ):
        # 最憨的神经网络,起手式
        super(Net , self).__init__()
        self.fc1 = nn.Linear(N_STATES , 50)
        self.fc1.weight.data.normal_(0 , 0.1) # 初始化权重
        self.out = nn.Linear(50 , N_ACTIONS)  # 输出N个动作的Q
        self.out.weight.data.normal_(0 , 0.1)

    def forward(self , x):
        # 全连接 + 激活 + 输出
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value

# 定义强化学习框架
class DQN(object):
    def __init__(self):
        self.eval_net , self.target_net = Net() , Net()

        self.learn_step_counter = 0   # 定期更新target_net
        self.memory_counter = 0       # 用来更新记忆库
        self.memory = np.zeros((MEMORY_CAPACITY , N_STATES * 2 + 2))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters() , lr = LR)         # 只用更新 eval_net
        self.loss_func = nn.MSELoss()

    def choose_action(self , x):
        x = torch.unsqueeze(torch.FloatTensor(x) , 0)                                   # 升维送入神经网络 shape:(1 , N_STATES)
        if np.random.uniform() < EPSILON:
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value , 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)     # 返回最大的 index
        else:
            action = np.random.randint(0 , N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self , s , a , r , s_):
        transition = np.hstack((s , [a , r] , s_))
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index , : ] = transition
        self.memory_counter += 1

    def learn(self):
        # 先检查是否需要更新神经网络参数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 从记忆库抽样
        sample_index = np.random.choice(MEMORY_CAPACITY , BATCH_SIZE)
        b_memory = self.memory[sample_index , : ]
        b_s = torch.FloatTensor(b_memory[ : , : N_STATES])
        # 动作的数据类型得是整数型
        b_a = torch.LongTensor(b_memory[ : , N_STATES : N_STATES + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[ : , N_STATES + 1 : N_STATES + 2])     # a , r 都是瘦高条子
        b_s_ = torch.FloatTensor(b_memory[ : , -N_STATES :])

        # 提取q_eval中动作对应的部分
        q_eval = self.eval_net(b_s).gather(1 , b_a)
        # 注意gather里面必须填整型数据,gather(1 , b_a)代表按列取 b_a 对应的 q_eval
        q_next = self.target_net(b_s_).detach()  # .detach() : 斩断target_net的反向传递,即不让target_net更新,detach:使分离.
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE , 1)          # reshape(batch , 1) ,变成瘦高条子
        # 科普：max(1)[0]中的0指最大值,如果是[1]就指最大值的索引
        loss =  self.loss_func(q_eval , q_target)

        # 反向传递更新神经网络参数
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def save(self):
        torch.save(self.eval_net, 'eval_net.pkl')
        print('模型保存完毕')



# 训练部分
dqn = DQN()
total_r = []
print('\n记忆库填装中...')
for i_episode in range(400): # 400条命,前面200条命平均玩10步就死...,从第200个episode开始学习
    s = env.reset()
    ep_r = 0
    while 1 : # 每进行一次操作,抽32个例子学习进化神经网络
        if i_episode > 390 : env.render()
        a = dqn.choose_action(s)

        # 对环境输入动作得到反馈
        s_ , r , done , info = env.step(a)

        # 修改一下reward
        x , x_dot , theta , theta_dot = s_ # 这里的 N_STATES包括4个特征
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2

        dqn.store_transition(s , a , r ,s_)

        ep_r += r
        # 记忆库装满了就开始学习
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            if done:
                print(
                    'Ep: ' , i_episode,
                    '|Ep_r: ' , round(ep_r , 2) # 保留两位
                )
                total_r.append(ep_r)
                break
        s = s_
dqn.save()
print('后100eps的平均reward为: ' , np.mean(total_r[-100: ])) # 549.9343623317139


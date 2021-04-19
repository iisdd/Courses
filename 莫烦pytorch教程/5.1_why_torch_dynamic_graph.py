"""
用一个RNN的例子:用sin预测cos,说明为什么torch是动态的
核心思想:torch可以接受动态变化的输入(输入的长度变化)
"""
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)

# 超参数
INPUT_SIZE = 1
LR = 0.02

class RNN(nn.Module):
    def __init__(self):
        super(RNN , self).__init__()

        self.rnn = nn.RNN(
            input_size= 1,
            hidden_size= 32,
            num_layers= 1,
            batch_first= True, # batch size作为第一个维度,(batch , time_step , input_size)
        )
        self.out = nn.Linear(32 , 1)

    def forward(self , x , h_state):
        # x (batch , time_step , input_size)
        # h_state (n_layers , batch , hidden_size)
        # r_out (batch , time_step , output_size)
        r_out , h_state = self.rnn(x , h_state)

        outs = []  # 这里可以反映出torch的动态性
        for time_step in range(r_out.size(1)):  # 也就是每一个time_step , r_out (batch , time_step , output_size)
            outs.append(self.out(r_out[ : , time_step , : ]))
        return torch.stack(outs , dim = 1) , h_state

rnn = RNN()
print(rnn)

optimizer = torch.optim.Adam(rnn.parameters() , lr = LR)
loss_func = nn.MSELoss()

h_state = None  # 相当于全给他初始化成0

plt.figure(1 , figsize=(12 , 5))
plt.ion()

####################以下部分体现区别###############################

#####################静态time steps################################
# for step in range(60):
#     start , end = step * np.pi ,(step+1)*np.pi
#     # 用sin预测cos , 多行注释:ctrl+/
#     steps = np.linspace(start , end , 10 , dtype = np.float32)

#####################动态time steps################################
# 即输入数量可以接受变化
step = 0
for i in range(60):
    dynamic_steps = np.random.randint(1 , 4)  # 随机的time_steps : 10 , 20 , 30
    start , end = step * np.pi , (step + dynamic_steps) * np.pi
    step += dynamic_steps

    steps = np.linspace(start , end , 10 * dynamic_steps , dtype = np.float32)

#########################以上部分有区别##############################
    print(len(steps))                   # 看输入RNN的是多少time step

    x_np = np.sin(steps)
    y_np = np.cos(steps)

    # 数据类型以及形状变化, shape(batch , time_step , input_size)
    x = torch.from_numpy(x_np[np.newaxis , : , np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis , : , np.newaxis])

    prediction , h_state = rnn(x , h_state)
    # 关键一步来了
    h_state = h_state.data              # h_state输出和输入要求的形状不同,要变化一下才不会报错

    loss = loss_func(prediction , y)    # cross entropy loss
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # plotting
    plt.plot(steps, y_np.flatten(), 'r-')
    plt.plot(steps, prediction.data.numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)

plt.ioff()
plt.show()


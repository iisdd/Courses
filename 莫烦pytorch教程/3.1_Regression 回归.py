# 总体流程:
# 1.搭建网络结构
# 2.设置optimizer, loss_func
# 3.for循坏算loss, 梯度清零 optimizer.zero_grad() -> 梯度回传 loss.backward() -> 优化参数 optimizer.step()

import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

#############第一种神经网络###########################
class Net(torch.nn.Module):                                 # 继承神经网络
    def __init__(self , n_feature , n_hidden , n_output):
        super(Net , self).__init__()
        self.hidden = torch.nn.Linear(n_feature , n_hidden) # 隐藏层神经元数
        self.predict = torch.nn.Linear(n_hidden , n_output)
        
    def forward(self , x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x
net = Net(1 , 10 , 1)
######################################################

##############第二种##############################
'''net = torch.nn.Sequential(
    torch.nn.Linear(1 ,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10 , 1),
    )'''
##############第二种##############################

print(net)

# 实例曲线拟合
# fake data
x = torch.unsqueeze(torch.linspace(-1 , 1 , 100) , dim = 1)
y = x.pow(2) + 0.2*torch.rand(x.size()) # x**2加上一些噪声
plt.ion()                               # 实时打印
plt.show()

optimizer = torch.optim.SGD(net.parameters() , lr = 0.2 ,momentum = 0.8)
# 优化器得分:
# SGD:0.0042
# Momentum:0.0030
# RMSprop:不收敛
# Adam:0.0042 折线哥
loss_func = torch.nn.MSELoss()          # 均方差mean square error

for t in range(300):
    prediction = net(x)
    loss = loss_func(prediction , y)

    optimizer.zero_grad()               # 梯度清零,防止爆炸
    loss.backward()                     # 误差反向传递
    optimizer.step()
    if t % 5 == 0:
        # 显示学习的过程
        plt.cla()
        plt.scatter(x.data.numpy() , y.data.numpy())
        plt.plot(x.data.numpy() , prediction.data.numpy() , 'r-' , lw = 5) # 线的宽度
        plt.text(0.5 , 0 , 'Loss = %.4f' % loss.data.numpy() , fontdict = {'size': 20 , 'color' : 'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

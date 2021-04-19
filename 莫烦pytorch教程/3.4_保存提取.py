# 保存: torch.save(model, 'model.pkl')
# 加载: model = torch.load('model.pkl')
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt

torch.manual_seed(1) # 可复现

# 假数据
x = torch.unsqueeze(torch.linspace(-1 , 1 , 100) , dim = 1)
y = x.pow(2) + 0.2 * torch.rand(x.size())

def save():
    # 建立网络
    net1 = torch.nn.Sequential(
        torch.nn.Linear(1 , 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10 , 1)
        )
    # 优化器
    optimizer = torch.optim.SGD(net1.parameters() , lr = 0.2)
    loss_func = torch.nn.MSELoss()

    # 训练
    for t in range(100):
        prediction = net1(x)
        loss = loss_func(prediction , y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    torch.save(net1 , 'net.pkl')                        # pickle储存
    torch.save(net1.state_dict() , 'net_params.pkl')    # 保存参数,矩阵
    # 出图
    plt.figure(1 , figsize = (10 , 3))
    plt.subplot(131)
    plt.title('Net1')
    plt.scatter(x.data.numpy() , y.data.numpy())        # 散点图
    plt.plot(x.data.numpy() , prediction.data.numpy() , 'r-' , lw = 5)

def restore_net():
    net2 = torch.load('net.pkl')
    prediction = net2(x)
    # 出图
    plt.figure(1 , figsize = (10 , 3))
    plt.subplot(132)
    plt.title('Net2')
    plt.scatter(x.data.numpy() , y.data.numpy())        # 散点图
    plt.plot(x.data.numpy() , prediction.data.numpy() , 'r-' , lw = 5)

def restore_params():                                   # 先建立一样的结构
    net3 = torch.nn.Sequential(
        torch.nn.Linear(1 , 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10 , 1)
        )
    net3.load_state_dict(torch.load('net_params.pkl'))  # 效率高于pkl, 存大型网络一般是存参数
    prediction = net3(x)
    # 出图
    plt.figure(1 , figsize = (10 , 3))
    plt.subplot(133)
    plt.title('Net3')
    plt.scatter(x.data.numpy() , y.data.numpy()) # 散点图
    plt.plot(x.data.numpy() , prediction.data.numpy() , 'r-' , lw = 5)
    plt.show()

# 调用功能
save()
restore_net()
restore_params()
    

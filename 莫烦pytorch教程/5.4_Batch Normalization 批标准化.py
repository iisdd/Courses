'''
本程序用的例子是曲线拟合
批标准化核心思想：1、把数据分成小批。 2、把数据标准化成合适的区间(线性区间,把变化不明显的区间去掉,例如tanh)里的均匀分布
BN被添加在全连接和激活层之间
'''
import torch
from torch import nn
from torch.nn import init
import torch.utils.data as Data
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(1)    # reproducible
np.random.seed(1)

# 超参数
N_SAMPLES = 2000
BATCH_SIZE = 64
EPOCH = 12
LR = 0.03
N_HIDDEN = 8
ACTIVATION = torch.tanh     # 使用tanh
# ACTIVATION = torch.relu   # 使用relu
B_INIT = -0.2               # 模拟一个很差的初始化

# 训练集
x = np.linspace(-7 , 10 , N_SAMPLES)[ : , np.newaxis]
noise = np.random.normal(0 , 2 , x.shape)
y = np.square(x) - 5 + noise

# 测试集
test_x = np.linspace(-7 , 10 , 100)[ : , np.newaxis]
noise = np.random.normal(0 , 2 , test_x.shape)
test_y = np.square(test_x) - 5 + noise

# 把训练集、测试集从numpy变成tensor
train_x , train_y = torch.from_numpy(x).float() , torch.from_numpy(y).float()
test_x , test_y = torch.from_numpy(test_x).float() , torch.from_numpy(test_y).float()

# 抽样训练部分
train_dataset = Data.TensorDataset(train_x , train_y)
train_loader = Data.DataLoader(dataset=train_dataset , batch_size=BATCH_SIZE , shuffle=True , num_workers = 2,)

# 可视化散点图
# plt.scatter(train_x.numpy(), train_y.numpy(), c='#FF9359', s=50, alpha=0.2, label='train')
# plt.legend(loc='upper left')
# plt.show()

class Net(nn.Module):
    def __init__(self , batch_normalization=False): # 默认网络不使用批标准化
        super(Net , self).__init__()
        self.do_bn = batch_normalization
        self.fcs = []
        self.bns = []
        self.bn_input = nn.BatchNorm1d(1 , momentum=0.5)  # 批标准层

        for i in range(N_HIDDEN):               # 搭建隐层和BN层,真的牛逼,动态搭建 !!!
            input_size = 1 if i == 0 else 10    # 第一层输入为 1 ,其他层输入 10
            fc = nn.Linear(input_size , 10)     # 好高级的动态搭建啊
            setattr(self, 'fc%i' % i, fc)       # 把fc的内容赋给fci
            self._set_init(fc)                  # 初始化参数torch.nn里的init
            self.fcs.append(fc)
            if self.do_bn:                      # 夹心饼干,一层全连接跟一层批标准
                # 用setattr把层变成class属性
                bn = nn.BatchNorm1d(10 , momentum=0.5)
                setattr(self,'bn%i'%i,bn)
                self.bns.append(bn)


        self.predict = nn.Linear(10 , 1)    # 输出层
        self._set_init(self.predict)        # 初始化参数


    def _set_init(self ,layer):
        init.normal_(layer.weight , mean = 0 , std = .1)
        init.constant_(layer.bias , B_INIT)

    def forward(self , x):
        pre_activation = [x]                    # 每次激活层前的输入
        if self.do_bn : x = self.bn_input(x)    # 判断要不要把输入标准化
        layer_input = [x]                       # 每次全连接层的输入
        for i in range(N_HIDDEN):               # N_HIDDEN指隐层的数量,不是神经元数
            x = self.fcs[i](x)
            pre_activation.append(x)
            if self.do_bn : x = self.bns[i](x)
            x = ACTIVATION(x)                   # 激活层
            layer_input.append(x)
        # 输出层
        out = self.predict(x)
        return out , layer_input , pre_activation

# 建立两个nets,一个有BN一个没有
nets = [Net(batch_normalization=False), Net(batch_normalization=True)]
# print(*nets)  # 打印网络结构

opts = [torch.optim.Adam(net.parameters() , lr = LR) for net in nets]
loss_func = torch.nn.MSELoss()

# 画图部分
def plot_histogram(l_in, l_in_bn, pre_ac, pre_ac_bn):
    for i, (ax_pa, ax_pa_bn, ax, ax_bn) in enumerate(zip(axs[0, :], axs[1, :], axs[2, :], axs[3, :])):
        [a.clear() for a in [ax_pa, ax_pa_bn, ax, ax_bn]]
        if i == 0:
            p_range = (-7, 10);the_range = (-7, 10)
        else:
            p_range = (-4, 4);the_range = (-1, 1)
        ax_pa.set_title('L' + str(i))
        ax_pa.hist(pre_ac[i].data.numpy().ravel(), bins=10, range=p_range, color='#FF9359', alpha=0.5);ax_pa_bn.hist(pre_ac_bn[i].data.numpy().ravel(), bins=10, range=p_range, color='#74BCFF', alpha=0.5)
        ax.hist(l_in[i].data.numpy().ravel(), bins=10, range=the_range, color='#FF9359');ax_bn.hist(l_in_bn[i].data.numpy().ravel(), bins=10, range=the_range, color='#74BCFF')
        for a in [ax_pa, ax, ax_pa_bn, ax_bn]: a.set_yticks(());a.set_xticks(())
        ax_pa_bn.set_xticks(p_range);ax_bn.set_xticks(the_range)
        axs[0, 0].set_ylabel('PreAct');axs[1, 0].set_ylabel('BN PreAct');axs[2, 0].set_ylabel('Act');axs[3, 0].set_ylabel('BN Act')
    plt.pause(0.01)

# 训练部分
if __name__ == '__main__':
    f , axs = plt.subplots(4 , N_HIDDEN + 1 , figsize = (10 , 5))
    plt.ion()
    plt.show()

    # 训练
    losses = [[] , []]

    for epoch in range(EPOCH):
        print('EPOCH: ' , epoch)
        layer_inputs , pre_acts = [] , []
        for net , l in zip(nets , losses):
            net.eval()  # 转化成评估模式,固定住参数,不让变
            pred , layer_input , pre_act = net(test_x)
            l.append(loss_func(pred , test_y).data.item())
            layer_inputs.append(layer_input)
            pre_acts.append(pre_act)
            net.train()   # 重新变回训练模式
        plot_histogram(*layer_inputs , *pre_acts)

        # 抽样训练参数
        for step , (b_x , b_y) in enumerate(train_loader):
            for net , opt in zip(nets , opts):  # 分别训练两个网络
                pred, _ , _  = net(b_x)
                loss = loss_func(pred , b_y)
                opt.zero_grad()
                loss.backward()
                opt.step()

    plt.ioff()

    # plot training loss
    plt.figure(2)
    plt.plot(losses[0], c='#FF9359', lw=3, label='Original')
    plt.plot(losses[1], c='#74BCFF', lw=3, label='Batch Normalization')
    plt.xlabel('step');plt.ylabel('test loss');plt.ylim((0, 2000));plt.legend(loc='best')

    # evaluation
    # set net to eval mode to freeze the parameters in batch normalization layers
    [net.eval() for net in nets]    # set eval mode to fix moving_mean and moving_var
    preds = [net(test_x)[0] for net in nets]
    plt.figure(3)
    plt.plot(test_x.data.numpy(), preds[0].data.numpy(), c='#FF9359', lw=4, label='Original')
    plt.plot(test_x.data.numpy(), preds[1].data.numpy(), c='#74BCFF', lw=4, label='Batch Normalization')
    plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='r', s=50, alpha=0.2, label='train')
    plt.legend(loc='best')
    plt.show()

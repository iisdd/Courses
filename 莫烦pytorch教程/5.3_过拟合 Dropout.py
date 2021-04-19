"""
这个程序用一个经过训练的神经网络在training data 和 test data上的表现来
对比普通方法和dropout的泛化能力。
解决过拟合的方法：
y = Wx
L1,L2..regularization:
1.L1正则化：cost = (Wx - real_y)^2 + abs(W)  # 不让W太复杂
2.L2正则化：cost = (Wx - real_y)^2 + (W)^2
3.神经网络专用的dropout regularization: 训练的时候随机的忽略一些神经元,使得这个神经网络不完整,
这样训练出来的结果就不会过于依赖某一个神经元。
加入dropout的方法也很简单,就是在搭神经网络的时候,在激活层前加一层dropout就行
"""
import torch
import matplotlib.pyplot as plt

#torch.manual_seed(1)

N_SAMPLES = 20
N_HIDDEN = 300

# 训练集
x = torch.unsqueeze(torch.linspace(-1 , 1 , N_SAMPLES) , 1)
# torch.normal(means, std, out=None) : 返回一个张量,包含从给定参数means,std的离散正态分布中抽取随机数
y = x + 0.3 * torch.normal(torch.zeros(N_SAMPLES , 1) , torch.ones(N_SAMPLES , 1))
# 总的来说就是斜率为1的线,上面的点有 ±0.3的浮动

# 测试集
test_x = torch.unsqueeze(torch.linspace(-1 , 1 , N_SAMPLES) , 1)
test_y = test_x + 0.3 * torch.normal(torch.zeros(N_SAMPLES , 1) , torch.ones(N_SAMPLES , 1))

# 可视化数据
plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.5, label='train')
plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.5, label='test')
plt.legend(loc='upper left')
plt.ylim((-2.5, 2.5))
plt.show()

# 憨憨过拟合网络
net_overfitting = torch.nn.Sequential(
    torch.nn.Linear(1 , N_HIDDEN),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN , N_HIDDEN),  # 这么简单的任务还要搞两层300的隐层...不overfitting才怪嘞
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN , 1),
)

# dropout网络
net_dropped = torch.nn.Sequential(
    torch.nn.Linear(1 , N_HIDDEN),
    torch.nn.Dropout(0.5), # 丢一半的神经元
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN , N_HIDDEN),
    torch.nn.Dropout(0.5),
    torch.nn.ReLU(),
    torch.nn.Linear(N_HIDDEN , 1),
)

# 打印网络结构
print(net_overfitting)
print(net_dropped)

optimizer_ofit = torch.optim.Adam(net_overfitting.parameters() , lr = 0.01)
optimizer_drop = torch.optim.Adam(net_dropped.parameters() , lr = 0.01)
loss_func = torch.nn.MSELoss()

plt.ion()

for t in range(500):
    pred_ofit = net_overfitting(x)
    pred_drop = net_dropped(x)
    loss_ofit = loss_func(pred_ofit , y)
    loss_drop = loss_func(pred_drop , y)

    optimizer_ofit.zero_grad()
    optimizer_drop.zero_grad()
    loss_ofit.backward()
    loss_drop.backward()
    optimizer_ofit.step()
    optimizer_drop.step()

    if t % 10 == 0:
        # 检测两个神经网络在测试集上的效果,先转成评估模式,这一步的目的是取消drop网络中的dropout概率
        net_overfitting.eval()
        net_dropped.eval()

        # 画图部分
        plt.cla()
        test_pred_ofit = net_overfitting(test_x)
        test_pred_drop = net_dropped(test_x)
        plt.scatter(x.data.numpy(), y.data.numpy(), c='magenta', s=50, alpha=0.3, label='train')
        plt.scatter(test_x.data.numpy(), test_y.data.numpy(), c='cyan', s=50, alpha=0.3, label='test')
        plt.plot(test_x.data.numpy(), test_pred_ofit.data.numpy(), 'r-', lw=3, label='overfitting')
        plt.plot(test_x.data.numpy(), test_pred_drop.data.numpy(), 'b--', lw=3, label='dropout(50%)')
        plt.text(0, -1.2, 'overfitting loss=%.4f' % loss_func(test_pred_ofit, test_y).data.numpy(), fontdict={'size': 20, 'color':  'red'})
        plt.text(0, -1.5, 'dropout loss=%.4f' % loss_func(test_pred_drop, test_y).data.numpy(), fontdict={'size': 20, 'color': 'blue'})
        plt.legend(loc='upper left'); plt.ylim((-2.5, 2.5));plt.pause(0.1)

        # 检测完了转回训练模式
        net_overfitting.train()
        net_dropped.train()

plt.ioff()
plt.show()

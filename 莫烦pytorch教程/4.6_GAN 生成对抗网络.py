"""
本次使用画画的例子来实现GAN,专业的画家画的画应该处在蓝色和橘色中间
"""
# 报错了：RuntimeError: one of the variables needed for gradient computation has been modified by
# an inplace operation: [torch.FloatTensor [128, 1]], which is output 0 of TBackward, is at version 2;
# expected version 1 instead. Hint: enable anomaly detection to find the operation that failed to compute its gradient,
# with torch.autograd.set_detect_anomaly(True). 不知道咋解决...
# 网上说可以1.下载0.3.1版本以下的torch,但是只有linux上能下
# 2.更换inplace操作,也就是把x += 1 换成 x = x + 1,但是我这也妹有吖
# 3.把所有inplace = True换成inplace = False,我这还是妹有吖
# 我 wnm嘞傻逼程序

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(1)
np.random.seed(1)

# 超参数部分
BATCH_SIZE = 64
LR_G = 0.0001       # Generate的 learning rate
LR_D = 0.0001       # Discriminator的 learning rate
N_IDEAS = 5         # 画画时的点子(输入的特征)
ART_COMPONENTS = 15 # 一共能在帆布上画几个点
PAINT_POINTS = np.vstack([np.linspace(-1 , 1 , ART_COMPONENTS)for _ in range(BATCH_SIZE)])

# show our beautiful painting range
# plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
# plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
# plt.legend(loc='upper right')
# plt.show()

def artist_works(): # 真人大师的画作(两条线之间)
    a = np.random.uniform(1 , 2 , size = BATCH_SIZE)[ : , np.newaxis]
    paintings = a * np.power(PAINT_POINTS , 2) + (a-1)
    paintings = torch.from_numpy(paintings).float()
    return paintings

# Generator
G = nn.Sequential(
    nn.Linear(N_IDEAS , 128),
    nn.ReLU(),
    nn.Linear(128 , ART_COMPONENTS),
)

# Discriminator
D = nn.Sequential(
    nn.Linear(ART_COMPONENTS , 128),
    nn.ReLU(),
    nn.Linear(128 , 1),
    nn.Sigmoid(),  # 只输出你这幅画是大师之作的概率
)

opt_D = torch.optim.Adam(D.parameters() , lr = LR_D)
opt_G = torch.optim.Adam(G.parameters() , lr = LR_G)

plt.ion()

for step in range(10000):
    artist_paintings = artist_works()           # 大师之作
    G_ideas = torch.randn(BATCH_SIZE , N_IDEAS) # roll 出 5个 ideas
    G_paintings = G(G_ideas)                    # 随机之作

    prob_artist0 = D(artist_paintings)          # D要尽量增大这个可能,判定大师之作good
    prob_artist1 = D(G_paintings)               # D要尽量减小这个可能,判定野鸡之作good

    # 这波啊,这波是左右互搏,养蛊,D要尽量区别开G和大师作来,G要尽量不让D区别出来
    # 所以才叫对抗生成网络吖
    D_loss = -torch.mean(torch.log(prob_artist0) + torch.log(1. - prob_artist1))
    G_loss = torch.mean(torch.log(1. - prob_artist1))  # 要减小G_loss就是要蒙的过D,让它判定野鸡作为大师作

    # 优化 D,G
    opt_D.zero_grad()
    D_loss.backward(retain_graph = True)        # reusing computational graph
    opt_D.step()

    opt_G.zero_grad()
    G_loss.backward()
    opt_G.step()

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings.data.numpy()[0], c='#4AD631', lw=3, label='Generated painting', )
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % prob_artist0.data.numpy().mean(),
                 fontdict={'size': 13})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -D_loss.data.numpy(), fontdict={'size': 13})
        plt.ylim((0, 3));
        plt.legend(loc='upper right', fontsize=10);
        plt.draw();
        plt.pause(0.01)

plt.ioff()
plt.show()

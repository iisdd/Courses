'''
卷积神经网络把图片越卷越厚(卷积),长宽越卷越小(池化),池化可以解决卷积时丢失信息的问题，
卷积只负责增加特征,压缩交给池化层。
结构一般是: 卷积层-激活层-池化层-...循环...-全连接层(分类)
比如最后输出4个类别,吃豆人的上下左右.
'''

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
from torch.autograd import Variable
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1) # 可复现

# 超参数
EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False  # 下过就不用下了

if not(os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    # 没下过就下一遍
    DOWNLOAD_MNIST = True
    
# 下载训练数据,手写字辨识
train_data = torchvision.datasets.MNIST(
    root = './mnist',
    train = True,
    transform = torchvision.transforms.ToTensor() , # 压缩 (0 - 255) -> (0 , 1)
    # 把数据类型转成tensor
    download = DOWNLOAD_MNIST
    )


'''
# 画一个例子
print(train_data.data.size())                 # (60000, 28, 28)
print(train_data.targets.size())               # (60000)
plt.imshow(train_data.data[0].numpy(), cmap='gray')
plt.title('%i' % train_data.targets[0])
plt.show()
# BIF改名了train_data -> data , train_labels -> targets
'''

# 小批次抽样的shape:(50, 1, 28, 28)
train_loader = Data.DataLoader(dataset = train_data , batch_size = BATCH_SIZE ,
                               shuffle = True)

# 抽两千个例子当test data
test_data = torchvision.datasets.MNIST(root = './mnist/' , train = False)
# 转换数据类型加标准化处理
test_x = torch.unsqueeze(test_data.test_data , dim = 1).type(torch.FloatTensor)[ : 2000]/255
# shape : (2000, 28 ,28) -> (2000 , 1 , 28 , 28)
test_y = test_data.test_labels[ : 2000]

# 建立CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN , self).__init__()
        # 第一层
        self.conv1 = nn.Sequential(  
            nn.Conv2d(                      # 卷积层, (1, 28, 28), 厚度放前面了
                in_channels = 1,            # 输入的厚度为1,如果是红绿蓝三色的就是3
                out_channels = 16,          # 16个filters , 相当于把图片分成16个部分来提取特征
                kernel_size = 5,            # filter的宽度为5个像素点
                stride = 1,                 # 滑动步长为1个像素点
                padding = 2,                # 边界填充2圈0
                ),                          # -> (16, 28, 28)
            nn.ReLU(),                      # 激活层  -> (16, 28, 28)
            nn.MaxPool2d(kernel_size = 2),  # 池化层,选2*2的最大值, -> (16, 14, 14)
            )
        # 第二层
        self.conv2 = nn.Sequential(         # (16, 14, 14)
            nn.Conv2d(16 , 32, 5 , 1 ,2),   # -> (32, 14, 14)
            nn.ReLU(),                      # -> (32, 14, 14)
            nn.MaxPool2d(2),                # -> (32, 7, 7)
            )
        # 输出层,全连接
        self.out = nn.Linear(32*7*7, 10)    # -> 10种手写数字分类

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)                   # (batch, 32, 7, 7)
        x = x.view(x.size(0) , -1)          # (batch, 32 * 7 * 7), view():降维
        output = self.out(x)
        return output , x

cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters() , lr = LR)
loss_func = nn.CrossEntropyLoss()

# 接下来是可视化过程
from matplotlib import cm
try: from sklearn.manifold import TSNE; HAS_SK = True
except: HAS_SK = False; print('Please install sklearn for layer visualization')
def plot_with_labels(lowDWeights , labels):
    plt.cla()
    X, Y = lowDWeights[ : , 0] , lowDWeights[ : , 1]
    for x, y, s in zip(X, Y, labels):
        c = cm.rainbow(int(255 * s / 9))
        plt.text(x, y, s, backgroundcolor=c, fontsize=9)
    plt.xlim(X.min(), X.max())
    plt.ylim(Y.min(), Y.max())
    plt.title('Visualize last layer')
    plt.show()
    plt.pause(0.01)
plt.ion()
# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):        # gives batch data, normalize x when iterate train_loader
        output = cnn(b_x)[0]                                # cnn output
        loss = loss_func(output, b_y)                       # cross entropy loss
        optimizer.zero_grad()                               # clear gradients for this training step
        loss.backward()                                     # backpropagation, compute gradients
        optimizer.step()                                    # apply gradients

        if step % 50 == 0:
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.2f' % accuracy)
            if HAS_SK:
                # Visualization of trained flatten layer (T-SNE)
                tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
                plot_only = 500
                low_dim_embs = tsne.fit_transform(last_layer.data.numpy()[:plot_only, :])
                labels = test_y.numpy()[:plot_only]
                plot_with_labels(low_dim_embs, labels)
plt.ioff()

# print 10 predictions from test data
test_output, _ = cnn(test_x[:10])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:10].numpy(), 'real number')


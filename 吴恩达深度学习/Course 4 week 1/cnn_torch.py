# 数据形状不对...,数据channel在第4位上,输入要求channel在第2位上(已解决)
# torch YYDS
import numpy as np
import matplotlib.pyplot as plt
import cnn_utils
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
import torchvision
torch.manual_seed(1)

# 创建文件夹model储存模型
if not os.path.exists('./model'):
    os.mkdir('./model')

# 超参
EPOCH = 200
BATCH_SIZE = 64
LR = 0.001



X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = cnn_utils.load_dataset()
np.random.seed(1)

# 数据处理,变成one-hot模式
X_train = X_train_orig/255.
X_train = torch.from_numpy(X_train)
# print(X_train.type()) # torch.DoubleTensor

print(X_train.shape)    # (1080, 64, 64, 3)
print(type(X_train))
X_train = X_train.reshape((-1, 3, 64, 64))
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = X_test_orig/255.
X_test = X_test.reshape((-1, 3, 64, 64))
X_test = torch.tensor(X_test, dtype=torch.float32)
Y_train = cnn_utils.convert_to_one_hot(Y_train_orig, 6).T
Y_train = torch.tensor(Y_train, dtype=torch.float32)
Y_test = cnn_utils.convert_to_one_hot(Y_test_orig, 6).T
print(Y_test.shape) # (120, 6)

# 建立CNN网络
class CNN(nn.Module):
    def __init__(self):
        super(CNN , self).__init__()
        # 第一层
        self.conv1 = nn.Sequential(
            nn.Conv2d(                      # 卷积层, (3, 64, 64)
                in_channels = 3,            # 输入的厚度为1,如果是红绿蓝三色的就是3
                out_channels = 8,           # 8个filters , 相当于把图片分成8个部分来提取特征
                kernel_size = 5,            # filter的宽度为5个像素点
                stride = 1,                 # 滑动步长为1个像素点
                padding = 2,                # 边界填充2圈0
                ),                          # -> (8, 64, 64)
            nn.ReLU(),                      # 激活层  -> (8, 64, 64)
            nn.MaxPool2d(kernel_size = 2),  # 池化层,选2*2的最大值, -> (8, 32, 32)
            )
        # 第二层
        self.conv2 = nn.Sequential(         # (8, 32, 32)
            nn.Conv2d(8 , 16, 5 , 1 ,2),    # -> (16, 32, 32)
            nn.ReLU(),                      # -> (16, 32, 32)
            nn.MaxPool2d(2),                # -> (16, 16, 16)
            )
        # 第三层
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),     # -> (32, 16, 16)
            nn.ReLU(),
            nn.MaxPool2d(4),                # -> (32, 4, 4)
        )
        # 输出层
        self.out = nn.Linear(32 * 4 * 4 , 6)# -> 6种手势

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)                   # (batch, 32, 4, 4)
        x = x.view(x.size(0) , -1)          # (batch, 32 * 4 * 4)
        output = self.out(x)
        return output , x

cnn = CNN()
print(cnn)

optimizer = torch.optim.Adam(cnn.parameters() , lr = LR)
loss_func = nn.MSELoss()

# train
costs = []
(m, n_H0, n_W0, n_C0) = X_train.shape
n_y = Y_train.shape[1]
seed = 3
for epoch in range(EPOCH):
    minibatch_cost = 0                      # 记录这一次遍历样本的平均误差
    num_minibatches = int(m / BATCH_SIZE)   # 整体样本分成了几个minibatch
    seed += 1
    minibatches = cnn_utils.random_mini_batches(X_train, Y_train, BATCH_SIZE, seed)  # 返回一个列表,里面是分出来的minibatch
    # 对每个minibatch进行一次训练
    for minibatch in minibatches:
        (minibatch_X, minibatch_Y) = minibatch
        # 训练&计算cost
        output = cnn(minibatch_X)[0]
        loss = loss_func(output, minibatch_Y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 累加成本值
        minibatch_cost += loss / num_minibatches

    if epoch % 5 == 0:
        print("当前是第 " + str(epoch) + " 代，成本值为：" + str(minibatch_cost))

    # 记录总成本走势
    costs.append(minibatch_cost)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per tens)')
plt.title("Learning rate =" + str(LR))
plt.show()

test_output, _ = cnn(X_test)
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y[:10], 'prediction number')
print(Y_test_orig[0][:10], 'real number')
print('准确率: ', np.sum(pred_y == Y_test_orig[0])/len(pred_y)) # 91.67
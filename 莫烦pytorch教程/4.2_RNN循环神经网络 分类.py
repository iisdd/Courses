'''
还是用那个识别手写数字的例子,把输入的图片拆成一条一条的,(从上到下)
这样就可以看做时间连续的输入了。
'''
import torch
from torch import nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

torch.manual_seed(1)    # 可复现

# 超参数
EPOCH = 1
BATCH_SIZE = 64
TIME_STEP = 28          # 图片分为 28行
INPUT_SIZE = 28         # 每一行输入 28个像素点
LR = 0.01
DOWNLOAD_MNIST = False

# Mnist数据集
train_data = dsets.MNIST(
    root = './mnist/',
    train = True ,                      # 训练集
    transform = transforms.ToTensor(),  # 把图片数据转化为tensor,(数量 x 高度 x 宽度)
    download = DOWNLOAD_MNIST,
)

print(train_data.data.size())           # (60000 , 28 , 28)
print(train_data.targets.size())        # 60000
# 展示一下第一个数据
plt.imshow(train_data.data[0].numpy() , cmap = 'gray')
plt.title('%i' % train_data.targets[0])
plt.show()

# 小批次抽样
train_loader = torch.utils.data.DataLoader(dataset=train_data , batch_size=BATCH_SIZE , shuffle = True)

# 抽样2000个测试集
test_data = dsets.MNIST(root='./mnist/', train=False, transform=transforms.ToTensor())
# 转换数据类型加标准化处理
test_x = test_data.data.type(torch.FloatTensor)[ :2000]/255
test_y = test_data.targets.numpy()[ :2000]


# 开始搭建 RNN部分
class RNN(nn.Module):
    def __init__(self):
        super(RNN , self).__init__()

        self.rnn = nn.LSTM(     # 如果用 nn.RNN(),学习速度很慢
            input_size=INPUT_SIZE,
            hidden_size=64,     # units的数量(开影分身共同做决定)
            num_layers=1,
            batch_first=True,
        )

        self.out = nn.Linear(64 , 10)

    def forward(self , x):
        r_out , (h_n , h_c) = self.rnn(x , None)  # None代表 0初始化 hidden state
        # h_n代表主线 hidden_state , h_c代表分线 hidden_state
        # x shape: (batch , time_step , input_size)
        # r_out shape: (batch , time_steo , output_size)
        # h_n,h_c shape:(n_layers , batch , hidden_size)

        # 选择最后一个 time_step的 r_out作为输出
        out = self.out(r_out[ : , -1 , : ])
        return out
# 打印 rnn结构
rnn = RNN()
print(rnn)
# 训练优化部分
optimizer = torch.optim.Adam(rnn.parameters() , lr = LR)
loss_func = nn.CrossEntropyLoss()
# 注意torch里的CrossEntropyLoss用的标签是真实标签,比如是数字7这个标签也就是7,而不是000000100(one-hot形式)

# 训练和测试部分
for epoch in range(EPOCH):
    for step, (b_x , b_y) in enumerate(train_loader):
        b_x = b_x.view(-1 , 28 , 28) # torch.view()相当于reshape() , -1代表交给机器来填
        # 比如这里的 b_x的shape 就是(batch , time_step , input_size)
        output = rnn(b_x)
        loss = loss_func(output , b_y) # cross entropy loss
        optimizer.zero_grad() # 梯度清零,防止梯度爆炸
        loss.backward()
        optimizer.step()

        if step % 50 == 0: # 每50步展示一下准确率
            test_output = rnn(test_x)
            pred_y = torch.max(test_output , 1)[1].data.numpy()
            accuracy = float((pred_y == test_y).astype(int).sum() / float(test_y.size))
            print('EPOCH: ' , epoch , '| train loss: %.4f' % loss.data.numpy() , '| test accuracy: %.2f' % accuracy)

# 打印训练后前10个数的测试结果
test_output = rnn(test_x[ : 10].view(-1 , 28 , 28))
pred_y = torch.max(test_output , 1)[1].data.numpy()
print(pred_y , 'prediction number')
print(test_y[ : 10] , 'real number')
# 当然训练结果完全比不上CNN,图像处理不是它的绝活嗷

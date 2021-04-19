"""
用CNN的例子来尝试GPU加速运算
核心思想:手动在tensor数据后面加.cuda()
"""
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision

torch.manual_seed(1)

EPOCH = 1
BATCH_SIZE = 50
LR = 0.001
DOWNLOAD_MNIST = False

train_data = torchvision.datasets.MNIST(root = './mnist/' , train = True ,transform = torchvision.transforms.ToTensor(),
                                        download = DOWNLOAD_MNIST , )
train_loader = Data.DataLoader(dataset=train_data , batch_size=BATCH_SIZE , shuffle = True)

test_data = torchvision.datasets.MNIST(root = './mnist/' , train = False)

#########重大改变嗷 , 在tensor类型数据后面加.cuda() , 把tensor送入GPU ########################
test_x = torch.unsqueeze(test_data.test_data , dim = 1).type(torch.FloatTensor)[ : 2000].cuda()/255
test_y = test_data.test_labels[ : 2000].cuda()

class CNN(nn.Module):
    def __init__(self):
        super(CNN , self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=1 , out_channels=16 , kernel_size=5 , stride=1 , padding=2,),
                                   nn.ReLU() , nn.MaxPool2d(kernel_size=2) , )
        # 输入厚度1、filters数量16(卷成16个特征)、filter像素宽度5、滑动步长1、边界填充2 、池化层宽度2*2
        self.conv2 = nn.Sequential(nn.Conv2d(16, 32 , 5 , 1 , 2) , nn.ReLU() , nn.MaxPool2d(2))
        self.out = nn.Linear(32 * 7 * 7 , 10)
        # (1,28,28)->卷积卷厚(16,28,28)->池化变窄(16,14,14)->卷积变厚(32,14,14)->池化变窄(32,7,7)

    def forward(self , x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0) , -1)
        output = self.out(x)
        return output


cnn = CNN()
# 把参数和命令都送到GPU里去
cnn.cuda()

optimizer = torch.optim.Adam(cnn.parameters() , lr = LR)
loss_func = nn.CrossEntropyLoss()   # 自带softmax

for epoch in range(EPOCH):
    for step,(x, y) in enumerate(train_loader):
        # 这里也要送入cuda,反正tensor都要送入cuda来计算
        b_x = x.cuda()
        b_y = y.cuda()

        output = cnn(b_x)
        loss = loss_func(output , b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            test_output = cnn(test_x)

            # 送入cuda
            pred_y = torch.max(test_output , 1)[1].cuda().data

            accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
            print('Epoch: ' , epoch , '|train loss: %.4f' % loss.data.cpu().numpy() ,
                  'test accuracy: %.2f' % accuracy)


# 展示前十个数的预测结果
test_output = cnn(test_x[ : 10])

pred_y = torch.max(test_output , 1)[1].cuda().data
print(pred_y , 'prediction number')
print(test_y[ : 10] , 'real number')

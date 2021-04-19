import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F

n_data = torch.ones(100 , 2)    # 100个假数据,列是坐标
x0 = torch.normal(2*n_data , 1) # tensor,shape = (100 , 2)
y0 = torch.zeros(100)           # tensor,shape = (100 , )
x1 = torch.normal(-2*n_data , 1)# tensor,shape = (100 , 1)
y1 = torch.ones(100)            # tensor,shape = (100 , )

x = torch.cat((x0 , x1) , 0).type(torch.FloatTensor)
y = torch.cat((y0 , y1) , ).type(torch.LongTensor)

# plt.scatter(x.data.numpy()[ : , 0] , x.data.numpy()[ : , 1] , c = y.data.numpy(),
#            s = 100 , lw = 0,cmap = 'RdYlGn')
# plt.show()

class Net(torch.nn.Module):                                 # 继承神经网络
    def __init__(self , n_feature , n_hidden , n_output):
        super(Net , self).__init__()
        self.hidden = torch.nn.Linear(n_feature , n_hidden) # 隐藏层神经元数
        self.predict = torch.nn.Linear(n_hidden , n_output)
        
    def forward(self , x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x

net = Net(2 , 10 , 2)   # 两个输入,(x,y)坐标,两个输出(分两类) [0,1]和[1,0]
print(net)

plt.ion()               # 实时画图
plt.show()

optimizer = torch.optim.SGD(net.parameters() , lr = 0.05)
loss_func = torch.nn.CrossEntropyLoss() # 自带一层softmax, 交叉熵用于one-hot的多分类
# 比如: 标签:[0,0,1], 实际:[0.1,0.2,0.7] 计算误差


for t in range(100):
    out = net(x)
    # print(out.shape, y.shape)
    loss = loss_func(out , y) # out还不是概率, [-10,20,30] -> softmax -> [0.1,0.2,0.7]

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if t % 2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out) , 1)[1]   # 过完这一道就成了概率
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[ : ,0] , x.data.numpy()[ : ,1] , c = pred_y , s = 100,
                    lw = 0,cmap = 'RdYlGn')
        accuracy = sum(pred_y == target_y)/200          # 准确率
        plt.text(1.5 , -4 , 'Accurracy = %.2f' % accuracy , fontdict = {'size':20 , 'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()

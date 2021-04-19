import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F


class Net(torch.nn.Module):                                     # 继承神经网络
    def __init__(self , n_feature , n_hidden , n_output):
        super(Net , self).__init__()
        self.hidden = torch.nn.Linear(n_feature , n_hidden)     # 隐藏层神经元数
        self.predict = torch.nn.Linear(n_hidden , n_output)
        
    def forward(self , x):
        x = torch.relu(self.hidden(x))
        x = self.predict(x)
        return x
# method1 直接定义层神经元数
net1 = Net(2 , 10 , 2)

# method2 一层一层累加神经层
net2 = torch.nn.Sequential(
    torch.nn.Linear(2 ,10),
    torch.nn.ReLU(),
    torch.nn.Linear(10 , 2),
    )
print(net1)
print(net2)

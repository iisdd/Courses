import torch
from torch.autograd import Variable     # 求梯度
import matplotlib.pyplot as plt

# fake data
x = torch.linspace(-5 , 5 , 200)        # -5到5取200个点的线
print(type(x), type(x.numpy()))

x_np = x.data.numpy()

# 激活后要转成np才能画
y_relu = torch.relu(x).data.numpy()
y_sigmoid = torch.sigmoid(x).data.numpy()
y_tanh = torch.tanh(x).data.numpy()
# y_softmax = torch.softmax(x).data.numpy()

# 看一下4个激励函数形状
# relu
plt.figure(1 , figsize = (8 , 6))
plt.subplot(221)
plt.plot(x_np , y_relu , c = 'red' , label = 'relu')
plt.ylim((-1 , 5))
plt.legend(loc = 'best')

# sigmoid
plt.subplot(222)
plt.plot(x_np , y_sigmoid , c = 'red' , label = 'sigmoid')
plt.ylim((-0.2 , 1.2))
plt.legend(loc = 'best')

# tanh
plt.subplot(223)
plt.plot(x_np , y_tanh , c = 'red' , label = 'tanh')
plt.ylim((-1.2 , 1.2))
plt.legend(loc = 'best')

# softmax有点问题,略过吧
'''
plt.subplot(224)
plt.plot(x_np , y_softmax , c = 'red' , label = 'softmax')
plt.ylim((-0.2 , 6))
plt.legend(loc = 'best')'''
plt.show()

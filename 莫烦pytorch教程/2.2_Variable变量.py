import torch
from torch.autograd import Variable

tensor = torch.FloatTensor([[1,2] , [3,4]])
variable = Variable(tensor , requires_grad = True)  # 梯度回传
print(tensor)
print(variable)
t_out = torch.mean(tensor * tensor)                 # 点乘,对应位置上乘出来
v_out = torch.mean(variable * variable)
# tensor不能反向传播,variable可以反向传播
print(t_out , v_out)
v_out.backward()
# v_out = 1/4*sum(var*var)
# d(v_out)/d(var) = 1/4*2*variable = variable/2
# 嚯嚯,智能求导
print(variable.grad)
print(variable)
print(variable.data)            # tensor
print(variable.data.numpy())    # numpy

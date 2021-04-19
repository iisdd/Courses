import torch
import numpy as np
np_data = np.arange(6).reshape((2,3))
torch_data = torch.from_numpy(np_data)
print(
    'numpy:\n',np_data,
    '\ntorch:\n',torch_data,
)
tensor2array = torch_data.numpy()
print(
    'tensor2array:\n' , tensor2array)
# 注意[]之间有逗号的就是2维的,否则就是1维的数据

data = [-1,-2,1,2]
tensor = torch.FloatTensor(data)    # tensor默认变成浮点数
print(
    '\nabs',
    '\nnumpy:' , np.abs(data),
    '\ntorch:' , torch.abs(tensor))
print(
    '\nsin',
    '\nnumpy:' , np.sin(data),
    '\ntorch:' , torch.sin(tensor))

# 矩阵乘
data = [[1,2] , [3,4]]
tensor = torch.FloatTensor(data)
print(
    '\nnumpy:' , np.matmul(data , data) ,
    '\ntorch:' , torch.mm(tensor , tensor)) # torch 的操作对象只能是tensor
# mm = matrix multiple

# 点乘(按位乘)
a = np.array([1,2,3,4])
tensor = torch.FloatTensor(a)
print(
    '\nnumpy:' , np.dot(a , a),
    '\ntorch:' , torch.dot(tensor,tensor)) # 点乘

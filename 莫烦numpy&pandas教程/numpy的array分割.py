import numpy as np
A = np.arange(12).reshape(3,4)
print(A)
print(np.split(A , 3 , axis = 0))       # 列向分割
print(np.split(A , 2 , axis = 1))       # 行向分割
print(np.array_split(A , 3 , axis = 1)) # 不等分割
print(np.vsplit(A , 3))                 # 列向分割
print(np.hsplit(A , 2)[0])              # 行向分割


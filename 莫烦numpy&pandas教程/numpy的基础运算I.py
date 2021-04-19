import numpy as np
a = np.array([10,20,30,40])
b = np.arange(4)
print(a, b)
c = 10*np.sin(a)
print(c)
print(b < 3)
a = np.array([[1,1],[0,1]])
b = np.arange(4).reshape(2,2)
print(a)
print(b)
c = a*b                     # 每个位置逐个相乘
c_dot = np.dot(a,b)         # 矩阵运算
print(c)
print(c_dot)
print(a.dot(b))             # 另一种写法
a = np.random.random((2,4))
print(a)
print(np.sum(a, axis = 1))  # 行求和,0列1行
print(np.min(a, axis = 0))  # 列最小
print(np.max(a, axis = 1))

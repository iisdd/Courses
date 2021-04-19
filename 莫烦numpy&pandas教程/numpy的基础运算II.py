import numpy as np
A = np.arange(2,14).reshape((3,4))
print(A)
print(np.argmin(A))         # 返回最小值的索引
print(np.argmax(A))         # 整体的索引,这里是11
print(np.mean(A))
print(np.average(A))
print(A.mean())
print(np.median(A))
print(np.cumsum(A))         # 累加
print(np.diff(A))           # 累差
print(np.nonzero(A))        # 输出所有非零值的对应行列
A = np.arange(14 , 2 , -1).reshape((3,4))
A.sort()                    # 每行排序,内容不变
print(A)
# 转置
print(np.transpose(A))
print(A.T)
# 自乘
print((A.T).dot(A))
print(A.dot(A.T))
print(np.clip(A , 5 , 9))   # 滤波,所有小于5的都变成5,所有大于9的都变成9
print(np.mean(A, axis = 1)) # 1: 对行求平均, 虽然返回看起来像一行, 但是可以理解为arrange没形状

import numpy as np
A = np.array([1,1,1])
B = np.array([2,2,2])
C = np.vstack((A , B))
print(C)                                            # vertical stack 上下合并
print(A.shape , C.shape)                            # A只是一个序列,只显示长度
D = np.hstack((A,B))                                # horizontal stack 左右合并
print(D , D.shape)
# 横向数列到纵向数列
print(A[np.newaxis , : ].shape)
print(A[ : ,np.newaxis].shape)
print(A[ : ,np.newaxis])
A = np.array([1,1,1])[ : ,np.newaxis]               # np.newaxis这句话加在列,这个序列就变成竖的
B = np.array([2,2,2])[ : ,np.newaxis]
print(np.hstack((A,B,A)))
C = np.concatenate((A , B , B , A ) , axis = 1)     # 0是列,1是行
print(C)

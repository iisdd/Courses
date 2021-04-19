import numpy as np
A = np.arange(3,15)
print(A)
print(A[3])
A = np.arange(3,15).reshape(3,4)
print(A)
print(A[1][1])
print(A[2,1])
print(A[1 , : ])
print(A[ : , 1])
print(A[1 , 1:2])
####################
for row in A:           # 打印行
    print(row)

for column in A.T:
    print(column)
print(A.flatten())      # 返回压扁后的A
for item in A.flat:     # A.flat是一个实体
    print(item)

# 展现python里的计算规则 -- broadcasting
import numpy as np
A = np.array([
    [56.0 , 0.0 , 4.4 , 68.0],
    [1.2 , 104.0 , 52.0 , 8.0],
    [1.8 , 135.0 , 99.0 , 0.9]
])
print(A,'\n')
cal = A.sum(axis = 0) # 按列求和,竖直相加
print(cal,'\n')
# 求不同食物中碳水化合物含量
percentage = 100 * A[0 , : ] / cal.reshape(1 , 4)
print(percentage)
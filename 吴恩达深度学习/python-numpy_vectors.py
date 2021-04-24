# numpy的基础语法
import numpy as np
a = np.random.randn(5)          # 产生5个高斯分布的随机数
# a = a.reshape(5 , 1)          # 改变结构方法1
# a = a[ : ,np.newaxis]         # 方法2:这个可以给他升维
print(a.shape)                  # 结果应该是(5 , )
# 不推荐使用这种秩为 1的数据结构,推荐使用以下表达
a = np.random.randn(5 , 1)      # 产生一个(5,1)的随机元素的矩阵
assert (a.shape == (5 , 1))     # a.shape如果不是(5,1)就会报错,异常诊断
print(a , '\n' , a.T)
print(np.dot(a , a.T))
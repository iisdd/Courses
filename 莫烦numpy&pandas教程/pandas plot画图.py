import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# plot data
# Series , 向量
data = pd.Series(np.random.randn(1000),index = np.arange(1000))
print(data)
data = data.cumsum()
print(data)
data.plot()
plt.show()

# DataFrame , 矩阵
data = pd.DataFrame(np.random.randn(1000,4),
                    index = np.arange(1000),
                    columns = list('ABCD'))
# 随机生成1000行4列的矩阵
data = data.cumsum()
print(data.head(3)) # 打印前几个数据
data.plot()
# plot methods:
# 'bar','hist','box','kde','area','scatter','hex','pie'
ax = data.plot.scatter(x='A',y='B',color='DarkBlue',label='Class 1')
data.plot.scatter(x = 'A',y = 'C',color='DarkGreen',label='Class 2',ax = ax)
# ax = ax 代表画在一个图里
plt.show()

import pandas as pd
import numpy as np
s = pd.Series([1,3,6,np.nan,44,1])
print(s)
dates = pd.date_range('20160101' , periods = 6)
print(dates)
df = pd.DataFrame(np.random.randn(6,4) , index = dates , columns = ['a','b','c','d'])
# df = pd.DataFrame(np.random.randn(6,4) , index=None, columns=None)
print(df)
df1 = pd.DataFrame(np.arange(12).reshape(3,4))
print(df1)
df2 = pd.DataFrame({'A':1.,
                    'B':pd.Timestamp('20130102'),
                    'C':pd.Series(1,index = list(range(4)),dtype = 'float32'),
                    'D':np.array([3]*4,dtype = 'int32'),
                    'E':pd.Categorical(['test','train','test','train']),
                    'F':'foo'}) # 自动补齐最长的
print(df2)
print(df2.index)        # 打印行名
print(df2.columns)      # 打印列名
print(df2.values)       # 打印所有元素,相当于原来的matrix去掉行列说明
print(df2.describe())   # 数字的统计,字符串的列不行
print(df2.T)            # 转置
df2 = df2.sort_index(axis = 1,ascending = False)    # 按列倒序(大->小)
print(df2)
df2 = df2.sort_index(axis = 0,ascending = False)    # 按行倒序
print(df2)
df2 = df2.sort_values(by = 'E')                     # 对E列排序(行序号也变了)
print(df2)

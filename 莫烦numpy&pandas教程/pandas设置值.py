# df[index] = ..., 一般index用df + 条件语句表示 Ex: df.A > 0
import numpy as np
import pandas as pd
dates = pd.date_range('20130101' , periods = 6)
df = pd.DataFrame(np.arange(24).reshape(6,4),index = dates,
                  columns = ['A','B','C','D'])
df.iloc[2,2] = 1111
print(df)
df.loc['20130101','B'] = 2222
print(df)
df[df.A>4] = 0
print(df)
df.A[df.A==0] = 1
print(df)
df.B[df.A==1] = 233
print(df)
# 格式: 操作对象[操作条件]
# 加一列
df['E'] = pd.Series([1,2,3,4,5,6] , index = df.index)
df['F'] = np.nan
print(df)

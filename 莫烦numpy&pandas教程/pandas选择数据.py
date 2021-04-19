import numpy as np
import pandas as pd
dates = pd.date_range('20130101',periods = 6)
df = pd.DataFrame(np.arange(24).reshape((6,4)) , index = dates ,
                  columns = ['A','B','C','D'])
print(df)
# 格式:pd.DataFrame(数据,index = 行名,columns = 列名)
# 列选取
print(df['A'] )
print(df.A)
# 行选取
print(df[0:3])
print(df['20130102':'20130104'])
# select by label:loc, 标签选择
print(df.loc['20130102'])
print(df.loc[ '20130102' ,['A','B']])
# select by position:iloc, 序号选择
print(df.iloc[3,1]) # 只用数字坐标, 不打名字
print(df.iloc[[1,3,5],1:3])
# ps. ix 已经弃用了
# Boolean indexing
print(df[df.A>8])

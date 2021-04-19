import numpy as np
import pandas as pd
dates = pd.date_range('20130101',periods = 6)
df = pd.DataFrame(np.arange(24).reshape((6,4)) , index = dates ,
                  columns = ['A','B' , 'C','D'])
df.iloc[0,1] = np.nan
df.iloc[1,2] = np.nan
print(df)
# dropna: 丢, fillna:补
print(df.dropna(axis = 0,how = 'any'))
# 0对应有NAN的那一行丢掉,1对应列
# any的意思就是出现就丢,对应的all就是一整行全为NAN才丢
print(df.dropna(axis = 1,how = 'any'))
print(df.dropna(axis = 0,how = 'all'))
print(df.fillna(value = '?'))       # 把NAN填入值,属于返回操作,不改变原对象
print(df)
print(df.isnull())                  # 判断是否位置上有数据丢失
print(np.any(df.isnull()) == True)  # 是否至少一个数据丢失

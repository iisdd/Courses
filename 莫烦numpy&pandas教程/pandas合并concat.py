import numpy as np
import pandas as pd
# 这篇基本上是SQL的内容
# concatenating
df1 = pd.DataFrame(np.ones((3,4))*0 , columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1 , columns = ['a','b','c','d'])
df3 = pd.DataFrame(np.ones((3,4))*2 , columns = ['a','b','c','d'])
print(df1)
print(df2)
print(df3)
# 合并
res = pd.concat([df1,df2,df3],axis = 0,ignore_index = True) # 0: 列向
# 0竖向合并,1横向合并,重新排序行名
print(res)

# join,['inner','outer']
df1 = pd.DataFrame(np.ones((3,4))*0 , columns = ['a','b','c','d'],index=[1,2,3])
df2 = pd.DataFrame(np.ones((3,4))*1 , columns = ['b','c','d','e'],index=[2,3,4])
print(df1)
print(df2)
res = pd.concat([df1,df2],join='outer')                     # outer就是没有的地方用NAN填充
print(res)
res = pd.concat([df1,df2],join='inner',ignore_index=True)   # inner就只保存大家都有的列
print(res)
# join_axis已经弃用
# res = pd.concat([df1,df2],axis=1,join_axes=[df1.index])

# append直接对某个df操作,在他的基础上添加,concat是把所有积木拼起来
df1 = pd.DataFrame(np.ones((3,4))*0 , columns = ['a','b','c','d'])
df2 = pd.DataFrame(np.ones((3,4))*1 , columns = ['b','c','d','e'])
res = df1.append(df2,ignore_index = True)
print(res)
s1 = pd.Series([1,2,3,4],index=['a','b','c','d'])
res = df1.append(s1,ignore_index = True)                    # 加一行,ignore_index必须加
print(res)


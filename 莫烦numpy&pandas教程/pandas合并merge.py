import numpy as np
import pandas as pd
# simple example
left = pd.DataFrame({'key':['K0','K1','K2','K3'],
                     'A':['A0','A1','A2','A3'],
                     'B':['B0','B1','B2','B3']})
# 'key'是列的名字
right = pd.DataFrame({'key':['K0','K1','K2','K3'],
                      'C':['C0','C1','C2','C3'],
                      'D':['D0','D1','D2','D3']})
print(left)
print(right)
res = pd.merge(left,right,on = 'key') # 融合
print(res)

# merging two df by key/keys.
# consider two keys
left = pd.DataFrame({'key1':['K0','K0','K1','K2'],
                     'key2':['K0','K1','K0','K1'],
                     'A':['A0','A1','A2','A3'],
                     'B':['B0','B1','B2','B3']})

right = pd.DataFrame({'key1':['K0','K1','K1','K2'],
                      'key2':['K0','K0','K0','K0'],
                      'C':['C0','C1','C2','C3'],
                      'D':['D0','D1','D2','D3']})
print(left)
print(right)
# 两个key对应的值不同该怎么合并
# how = ['inner','outer','left','right']
res = pd.merge(left,right,on = ['key1','key2'],how = 'inner')
# 找相同的key合并,有点像两个for循环找相同
print(res)
res = pd.merge(left,right,on = ['key1','key2'],how = 'outer')
# 没有的数据用NaN代替
print(res)
l = pd.DataFrame({'key1':['K0','K0','K1','K2'],
                     'key2':['K0','K1','K0','K1'],
                     'A':['A0','A1','A2','A3'],
                     'B':['B0','B1','B2','B3']})

r = pd.DataFrame({'key1':['K0','K1','K1','K2'],
                      'key2':['K0','K0','K0','K0'],
                      'C':['C0','C1','C2','C3'],
                      'D':['D0','D1','D2','D3']})
res = pd.merge(l,r,on = ['key1','key2'],how = 'right')
# 这个right不是变量名,是个方法名嗷,即以右边那个df为主
print(res)

# indicator
df1 = pd.DataFrame({'col1':[0,1],'col_left':['a','b']})
df2 = pd.DataFrame({'col1':[1,2,2],'col_right':[2,2,2]})
print(df1)
print(df2)
res = pd.merge(df1,df2, on='col1' ,how = 'outer' , indicator = True)
# indicator显示是左边有还是右边有还是都有
res = pd.merge(df1,df2, on='col1' ,how = 'outer' , indicator = 'merge方式')
print(res)

# 格式：pd.merge(df1,df2,on = '合并的列名', how = '合并的依据',indicator = True)
# merged by index
# 先定义df的index,即行的名字
left = pd.DataFrame({'A':['A0','A1','A2'],
                     'B':['B0','B1','B2']},
                    index = ['K0','K1','K2'])
right = pd.DataFrame({'C':['C0','C2','C3'],
                      'D':['D0','D2','D3']},
                     index = ['K0','K2','K3'])
print(left)
print(right)
# left_index,right_index默认为False
res = pd.merge(left,right,left_index = True,right_index = True,how = 'outer')
print(res)

# handle overlapping 重叠
boys = pd.DataFrame({'k':['K0','K1','K2'],'age':[1,2,3]})
girls = pd.DataFrame({'k':['K0','K0','K3'],'age':[4,5,6]})
print(boys)
print(girls)
# suufixes加后缀
res = pd.merge(boys,girls,on='k',suffixes=['_boy','_girl'],how = 'outer')
print(res)


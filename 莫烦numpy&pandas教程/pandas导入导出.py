# 读取还可以设置header=False, index_col=False
import pandas as pd
data = pd.read_csv('student.txt')
print(data)
data.to_pickle('student.pickle')

'''
0. 要求实现一个功能与 reversed() 相同（内置函数 reversed(seq) 是返回一个迭代器，
是序列 seq 的逆序显示）的生成器。
例如：
>>> for i in myRev("FishC"):
    print(i, end='')

ChsiF
我的答案：
def myRev(string):
    length = len(string)
    pos = 0
    while 1 :
        pos -= 1
        yield string[pos]
        if pos == -(length):
            break

for i in myRev('FishC'):
    print(i , end = '')
'''
# 参考答案，太猛了
def myRev(data):
    #   起点是最右边的元素
    for index in range(len(data) - 1 , -1 , -1):#最右边的-1是倒着走的意思
    #   中间那个-1是最后一个元素，range不计右边所以终点是到第一个元素
        yield data[index]

for i in myRev('FishC'):
    print(i , end = '')

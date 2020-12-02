'''
2. 要求自己写一个 MyRev 类，功能与 reversed() 相同
（内置函数 reversed(seq) 是返回一个迭代器，是序列 seq 的逆序显示）。
例如：
>>> myRev = MyRev("FishC")
>>> for i in myRev:
    print(i, end='')

ChsiF
'''

class MyRev:
    def __init__(self , string):
        self.string = string
        self.pos = 0
        self.length = len(self.string)

    def __iter__(self):
        return self

    def __next__(self):
        self.pos -= 1
        if self.pos < -(self.length):
            raise StopIteration
        return self.string[self.pos]
    

myRev = MyRev("FishC.com")
for i in myRev:
    print(i, end='')

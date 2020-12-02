'''1. 按要求编写描述符 Record：记录指定变量的读取和写入操作，
并将记录以及触发时间保存到文件：record.txt
程序实现如下：
>>> class Test:
        x = Record(10, 'x')
        y = Record(8.8, 'y')

>>> test = Test()
>>> test.x
10
>>> test.y
8.8
>>> test.x = 123
>>> test.x = 1.23
>>> test.y = "I love FishC.com!"
'''
import time as t


def shijian():    
    return (t.asctime())


    str1 = ''
    with open('record.txt' , 'r') as r:
        for each in r:
            str1 += each
    return str1
    
class Record:
    def __init__(self , value = None , name = None):
        self.value = value
        self.name = name

    def __get__(self , instance , owner):
        with open('record.txt' , 'a') as r:                
            r.writelines(self.name + ' 变量于北京时间 '+
                         shijian()+' 被读取，'+self.name+' = '+
                         str(self.value)+'\n')
        return self.value

    def __set__(self , instance , value):
        self.value = value
        with open('record.txt' , 'a') as r:                
            r.writelines(self.name + ' 变量于北京时间 '+
                         shijian()+' 被修改，'+self.name+' = '+
                         str(self.value)+'\n')

class Test:
    x = Record(10 , 'x')
    y = Record(8.8 , 'y')

test = Test()

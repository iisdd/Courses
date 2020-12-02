'''2. 再来一个有趣的案例：编写描述符 MyDes，使用文件来存储属性，
属性的值会直接存储到对应的pickle（腌菜，还记得吗？）的文件中。
如果属性被删除了，文件也会同时被删除，属性的名字也会被注销
举个栗子：
>>> class Test:
        x = MyDes('x')
        y = MyDes('y')
        
>>> test = Test()
>>> test.x = 123
>>> test.y = "I love FishC.com!"
>>> test.x
123
>>> test.y
'I love FishC.com!'

产生对应的文件存储变量的值：

如果我们删除 x 属性：
>>> del test.x
>>>
对应的文件也不见了：
'''
import pickle
import os
def save_file(file_name , num):
    with open(file_name + '.pkl' , 'wb') as f:
        pickle.dump(num , f)


class MyDes:
    def __init__(self , name):
        self.name = name

    def __get__(self , instance , owner):
        return self.value

    def __set__(self , instance , value):
        self.value = value
        save_file(self.name , self.value)

    def __delete__(self , instance):
        os.remove(self.name + '.pkl')
        del self.name


class Test:
    x = MyDes('x')
    y = MyDes('y')

test = Test()

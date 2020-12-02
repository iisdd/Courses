'''0. 按要求编写描述符 MyDes：当类的属性被访问、修改或设置的时候，分别做出提醒。
程序实现如下：
>>> class Test:
        x = MyDes(10, 'x')

>>> test = Test()
>>> y = test.x
正在获取变量： x
>>> y
10
>>> test.x = 8
正在修改变量： x
>>> del test.x
正在删除变量： x
噢~这个变量没法删除~
>>> test.x
正在获取变量： x
8
'''
class MyDes:
        def __init__(self , value = None , name = None):
                self.value = value
                self.name = name

        def __get__(self , instance , owner):
                print('正在获取变量：%s' %self.name)
                return self.value

        def __set__(self , instance , value):
                print('正在修改变量：%s' %self.name)
                self.value = value

        def __delete__(self , instance):
                print('正在删除变量：%s' %self.name)
        
                print('嗷~这个变量没法删除~')
       

class Test:
        x = MyDes(10 , 'x')
test = Test()

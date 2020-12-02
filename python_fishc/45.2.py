'''2.编写一个 Counter 类，用于实时检测对象有多少个属性
程序实现如下：
>>> c = Counter()
>>> c.x = 1
>>> c.counter
1
>>> c.y = 1
>>> c.z = 1
>>> c.counter
3
>>> del c.x
>>> c.counter
2

我的答案：
class Counter:
    def __init__(self):
        self.counter = 0

    def __setattr__(self , name , value):
        if name != 'counter':
            super().__setattr__(name , value)
            self.counter += 1
        else:
            super().__setattr__('counter', value)

    def __delattr__(self , name):
        
        super().__delattr__(name)
        self.counter -= 1
'''
# 网站答案:(super就完事了)
class Counter:
    def __init__(self):
        super().__setattr__('counter' , 0)

    def __setattr__(self , name , value):
        super().__setattr__('counter' , self.counter + 1)
        super().__setattr__(name , value)

    def __delattr__(self , name):
        super().__setattr__('counter' , self.counter - 1)
        super().__delattr__(name)

c = Counter()

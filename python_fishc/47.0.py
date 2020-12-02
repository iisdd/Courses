'''
0. 根据课堂上的例子，定制一个列表，同样要求记录列表中每个元素被访问的次数。
这一次我们希望定制的列表功能更加全面一些，
比如支持 append()、pop()、extend() 原生列表所拥有的方法。你应该如何修改呢？

要求1：实现获取、设置和删除一个元素的行为（删除一个元素的时候对应的计数器也会被删除）

要求2：增加 counter(index) 方法，返回 index 参数所指定的元素记录的访问次数

要求3：实现 append()、pop()、remove()、insert()、clear() 和 reverse() 方法
（重写这些方法的时候注意考虑计数器的对应改变）

今天只有一道动动手的题目，但在写代码的时候要时刻考虑到你的列表增加了计数器功能，
所以请务必要考虑周全再提交答案。
'''

#  方法属性全继承，只用管count就行

class CountList(list):
    def __init__(self , *args):
        super().__init__(args)
        self.count = []       # 要用列表打败列表!
        for i in args:
            self.count.append(0)
            
    def __repr__(self):
        return super().__repr__()

    def __len__(self):
        return len(self.count)

    def __getitem__(self , key):
        self.count[key] += 1
        return super().__getitem__(key)

    def __setitem__(self , key , value):
        self.count[key] += 1
        super().__setitem__(key , value)
        
    def __delitem__(self , key):      
        del self.count[key]
        super().__delitem__(key)

    def counter(self , key):
        return self.count[key]
        
    def append(self , value):
        self.count.append(0)
        super().append(value)

    def pop(self , key = -1):
        del self.count[key]
        super().pop(key)

    def remove(self , value):
        key = super().index(value)
        del self.count[key]
        super().remove(value)

    def insert(self , key , value):
        self.count.insert(key , 0)
        super().insert(key , value)

    def clear(self):
        self.count.clear()
        super().clear()

    def reverse(self):
        self.count.reverse()
        super().reverse()
        

            
c = CountList(1 , 2 , 3 , 4)

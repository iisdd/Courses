'''
0. 问大家一个问题：Python 支持常量吗？相信很多鱼油的答案都是否定的，
但实际上 Python 内建的命名空间是支持一小部分常量的，比如我们熟悉的
True，False，None 等，只是 Python 没有提供定义常量的直接方式而已。
那么这一题的要求是创建一个 const 模块，功能是让 Python 支持常量。

说到这里大家可能还是一头雾水，没关系，我们举个栗子。

test.py 是我们的测试代码，内容如下：

# const 模块就是这道题要求我们自己写的
# const 模块用于让 Python 支持常量操作
import const

const.NAME = "FishC"
print(const.NAME)

try:
    # 尝试修改常量
    const.NAME = "FishC.com"
except TypeError as Err:
    print(Err)

try:
    # 变量名需要大写
    const.name = "FishC"
except TypeError as Err:
    print(Err)

执行后的结果是：
>>> 
FishC
常量无法改变！
常量名必须由大写字母组成！

在 const 模块中我们到底做了什么，使得这个模块这么有“魔力”呢？
大家跟着小甲鱼的提示，一步步来做你就懂了：
提示一：我们需要一个 Const 类
提示二：重写 Const 类的某一个魔法方法，指定当实例对象的属性被修改时的行为
提示三：检查该属性是否已存在
提示四：检查该属性的名字是否为大写
提示五：细心的鱼油可能发现了，怎么我们这个 const 模块导入之后就把它当对象来使用
（const.NAME = "FishC"）了呢？难道模块也可以是一个对象？
没错啦，在 Python 中无处不对象，到处都是你的对象。
使用以下方法可以将你的模块与类 A 的对象挂钩。

sys.modules 是一个字典，它包含了从 Python 开始运行起，被导入的所有模块。键就是模块名，值就是模块对象。

import sys
sys.modules[__name__] = A()
'''
class Const:

    def __setattr__(self , name , value):
        if hasattr(self , name) :
            print('常量无法改变！')
        elif not str(name).isupper():
            print('常量名必须由大写字母组成！')

        else:
            super().__setattr__(name , value)
            

import sys
sys.modules[__name__] = Const()















    
    

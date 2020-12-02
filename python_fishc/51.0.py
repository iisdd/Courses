'''
0. 执行下边 a.py 或 b.py 任何一个文件，都会报错，请改正程序。
注：这道题原理跟上一节课的课后作业（测试题 4、5）类似，如果上节课你搞懂了，
这道题应该可以想出解决方案，不要轻易看答案，除非你已经抓破头皮……

# a.py
import b

def x():
    print('x')

b.y()

# b.py
import a

def y():
    print('y')

a.x()

执行 b.py 引发下边异常：
>>> 
Traceback (most recent call last):
  File "/Users/FishC/Desktop/b.py", line 1, in <module>
    import a
  File "/Users/FishC/Desktop/a.py", line 1, in <module>
    import b
  File "/Users/FishC/Desktop/b.py", line 6, in <module>
    a.x()
AttributeError: 'module' object has no attribute 'x'
'''
#  改在实验夹里了

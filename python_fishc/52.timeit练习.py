#    介绍小段代码的引入
import timeit as t
#    1.用s字符串来代替函数
s ='''
try:
    str.__bool__

except AttributeError:
    pass
'''
print('try法时间：' ,t.timeit(stmt = s , number = 100000))

s ='''
if hasattr(str , '__bool__'):
    pass
'''

print('hasattr法时间：', t.timeit(stmt = s , number = 100000))

#    2.定义函数并setup
def test():
    '''测试函数'''
    l = [i for i in range(100)]

if __name__ == '__main__':
    print('列表生成时间:' ,t.timeit('test()' , setup = 'from __main__ import test'
                              , number = 100000))

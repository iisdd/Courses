'''
0. 编写一个函数power()模拟内建函数pow()，即power(x, y)为计算并返回x的y次幂的值。
'''
def power(x , y):
    result = 1
    for i in range(y):
        result *= x
    return result

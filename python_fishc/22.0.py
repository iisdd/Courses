'''
0. 使用递归编写一个 power() 函数模拟内建函数 pow()，即 power(x, y)
为计算并返回 x 的 y 次幂的值。
'''
def power(x ,y):
    if y == 1:
        return x
    else:
        return power(x, y - 1) * x

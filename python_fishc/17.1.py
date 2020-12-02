'''
1. 编写一个函数，利用欧几里得算法（脑补链接）求最大公约数，
例如gcd(x, y)返回值为参数x和参数y的最大公约数。
'''
def gcd(x , y):
    while x % y:
        x , y = y , x % y
    return y

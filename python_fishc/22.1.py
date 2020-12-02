'''
1. 使用递归编写一个函数，利用欧几里得算法求最大公约数，
例如 gcd(x, y) 返回值为参数 x 和参数 y 的最大公约数。
'''
def gcd(x , y):
    if x % y == 0:
        return y
    else:
        return gcd(y , x % y)

print(gcd(196 , 357))

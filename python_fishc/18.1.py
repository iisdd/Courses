'''
1. 寻找水仙花数
    
题目要求：如果一个3位数等于其各位数字的立方和，则称这个数为水仙花数。
例如153 = 1^3+5^3+3^3，因此153是一个水仙花数。编写一个程序，找出所有的水仙花数。
'''
def iswater(n):
    sum1 = 0
    target = n
    while n:
        sum1 += (n % 10) ** 3
        n //= 10
    if sum1 == target:
        return True

for i in range(100 , 1000):
    if iswater(i):
        print(i)

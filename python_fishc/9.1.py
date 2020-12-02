'''
1. 编写一个程序，求 100~999 之间的所有水仙花数。
   
如果一个 3 位数等于其各位数字的立方和，则称这个数为水仙花数。
例如：153 = 1^3 + 5^3 + 3^3，因此 153 就是一个水仙花数。
'''
def issx(num):
    sum = 0
    temp = num
    while temp:
        sum += (temp % 10) ** 3
        temp //= 10
    if sum == num:
        return True

for each in range(100 , 1000):
    if issx(each):
        print(each , end = ' ')
        

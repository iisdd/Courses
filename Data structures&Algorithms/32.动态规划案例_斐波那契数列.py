# 动态规划经典案例: 斐波那契数列
# 默认n > 2
def fibnacci(n):
    if n == 0:
        return 0
    if n==1 or n==2:
        return 1
    else:
        return fibnacci(n-1) + fibnacci(n-2)

def fibnacci_no_rec(n):
    f = [0, 1, 1]
    if n > 2:
        for i in range(n-2):
            f.append(f[-1]+f[-2])
    return f[n]

print(fibnacci(10))
print(fibnacci_no_rec(10))





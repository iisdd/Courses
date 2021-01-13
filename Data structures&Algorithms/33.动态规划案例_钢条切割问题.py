# 不同长度的的钢条有不同的价格
# 现有长度为n的钢条,问如何切割可使得总收益最大
import time

def cal_time(func):
    def wrapper(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print('%s running time: %s secs.' % (func.__name__, t2 - t1))
        return result
    return wrapper

p = [0, 1, 5, 8, 9, 10, 17, 17, 20, 24, 30]


def cut_rod_recurision_1(p, n):                                   # 用递归的方法求最大收益,慢慢慢
    if n == 0:
        return 0
    else:
        if n < len(p):
            res = p[n]
        else:
            res = 0
        for i in range(1, n):
            res = max(res, cut_rod_recurision_1(p, i) + cut_rod_recurision_1(p, n-i))
        return res

@cal_time
def c1(p, n):                                                     # 防止递归一直产生碎的时间段
    return cut_rod_recurision_1(p, n)

print(c1(p, 15))



def cut_rod_recurision_2(p, n):                                   # 改进版递归,不像上面递归两次,卡住一边,递归一次即可
    if n == 0:
        return 0
    else:
        if n < len(p):
            res = p[n]
        else:
            res = 0
        for i in range(1, min(n, len(p))):                        # 不能把钢材用到负数
            res = max(res, p[i] + cut_rod_recurision_2(p, n-i))
        return res

@cal_time
def c2(p, n):
    return cut_rod_recurision_2(p, n)

print(c2(p, 15))

# 递归自顶向下解决问题,需要重复计算子问题,时间复杂度高,引入动态规划可以自底向上解决问题
# 把计算过的子问题存起来,用已知的小问题来计算大问题

@cal_time
def cut_rod_dp(p, n):
    dp = [0]
    for i in range(1, n+1):                                       # dp增加的长度
        if i < len(p):
            res1 = p[i]
            res2 = i
        else:
            res1 = 0
            res2 = 0
        for j in range(1, min(len(p), i)):
            if p[j] + dp[i-j][0] > res1:
                res1 = p[j] + dp[i-j][0]
                res2 = j
        dp.append((res1, res2))
    print(dp)
    i = n
    cut = []
    while i > 0:
        cut.append(dp[i][1])
        i -= dp[i][1]
    print('切割方案: ', cut)
    return dp[-1][0]

print(cut_rod_dp(p, 15))


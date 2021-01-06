# 贪心算法经典案例: 找零问题
t = [100, 50, 20, 10, 5, 1]                     # 钞票面额

def change(t, n):                               # 给定钞票面额,需要找零的大小,返回各面值钞票张数(列表)
    m = [0 for _ in range(len(t))]
    for i in range(len(t)):
        m[i] = n//t[i]                          # 啥时候能有ti看吖
        n %= t[i]
    return m

print(change(t, 388))







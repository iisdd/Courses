# 求两个字符串的最长公共子序列,不需要连续的也算是子序列
# 对于m行n列的问题,往往需要m+1行n+1列的dp表
def lcs_length(x, y):
    m = len(x)
    n = len(y)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if x[i-1] == y[j-1]:                        # i,j分别表示包含了多少位x,y, 所以要-1
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    for _ in dp:                                        # 逐行打印小技巧
        print(_)
    return dp[-1][-1]

x = 'ABCBDAB'
y = 'BDCABA'
print(lcs_length(x, y))

def lcs(x, y):                                          # 功能扩充,增加返回公共子序列功能
    m = len(x)
    n = len(y)
    dp = [[0 for _ in range(n+1)] for _ in range(m+1)]
    last = [[0 for _ in range(n+1)] for _ in range(m+1)]# 上一节点位置,1:左上, 2:上, 3:左
    for i in range(1, m+1):
        for j in range(1, n+1):
            if x[i-1] == y[j-1]:                        # i,j分别表示包含了多少位x,y, 所以要-1
                dp[i][j] = dp[i-1][j-1] + 1
                last[i][j] = 1
            else:
                if dp[i-1][j] >= dp[i][j-1]:
                    dp[i][j] = dp[i-1][j]
                    last[i][j] = 2
                else:
                    dp[i][j] = dp[i][j-1]
                    last[i][j] = 3

    print('dp表:')
    for _ in dp:                                        # 逐行打印小技巧
        print(_)
    print('来处:')
    for _ in last:
        print(_)
    res = ''                                            # 最长的子序列
    i = m
    j = n
    while last[i][j] > 0:
        cur_dp = dp[i][j]
        if last[i][j] == 1:
            i -= 1
            j -= 1
            res = x[i] + res
        elif last[i][j] == 2:
            i -= 1
        else:
            j -= 1
    print('最长子序列: ', res)
    return dp[-1][-1]

x = 'ABCBDAB'
y = 'BDCABA'
print('最长子序列长度: ', lcs(x, y))

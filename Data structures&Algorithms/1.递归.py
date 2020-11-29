# 复杂度O(n!)
def hanoi(n, a, b, c): # n个盘子从a经过b移动到c
    if n == 1: # 结束标志
        print('%s->%s' % (a, c))
    else: # 调用自身
        # 先把上面n-1个盘子从a经过c移动到b
        hanoi(n - 1, a, c, b)
        # 再把最底下一个盘子从a经过b移动到c
        hanoi(1, a, b, c)
        # 再把n-1个盘子从b经过a移动到c
        hanoi(n - 1, b, a, c)

hanoi(3, 'a', 'b', 'c')

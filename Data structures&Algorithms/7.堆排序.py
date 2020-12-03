# 时间复杂度: O(nlogn),真的很复杂...还要多看几遍

# 先写一个向下调整的函数
def sift(li, low, high):
    '''
    :param li:   列表
    :param low:  根节点位置
    :param high: 最尾端的叶子节点位置
    :return:     不返回值,把li变成堆(有序)
    '''
    i = low                     # 父节点
    j = i*2 + 1                 # 左孩子节点
    tmp = li[low]               # 把原来的堆顶(省长)记录下来
    while j < high:             # 还有孩子节点就一直循环
        # 先从孩子节点里挑出最强的来
        if j+1 < high and li[j+1] > li[j]:
            j = j+1             # j指向右孩子节点
        if li[j] > tmp:         # 下级比领导还强,换人
            li[i] = li[j]
            i = j
            j = i*2 + 1
        else:                   # 贬官到此为止
            break
    li[i] = tmp                 # 有两种可能:1.break出来的,tmp就在i这个为止当官正好. 2.j > high了,tmp一蹦到底当村民.

def heap_sort(li):                      # 步骤: 1.把列表建成一个堆,具体过程:农村包围城市,从下往上sift  2.挨个出数
    n = len(li)
    for i in range((n-2)//2, -1, -1):
        # 倒着来,n-2 = (n-1)-1,n-1是最后一个叶子的下标,再减1是从孩子节点推父节点的公式
        sift(li, i, n-1)                # 这里的n-1是一个作弊的写法,但是不影响结果,本身high这个参数就是判断是否为叶子节点,如果不是那孩子节点也不会超过high
    # 建堆完成
    # print(li)
    for i in range(n-1, -1, -1):        # 开始排序,由于是原地排序,挨个出来的数会放到原来high的位置,然后high-1
        li[0], li[i] = li[i], li[0]     # 把最大的丢到high去
        sift(li, 0, i-1)                # 有序区就不参与向下调整了
        # print(li)

import numpy as np
li = [np.random.randint(0, 100) for _ in range(20)]
# print(li)
heap_sort(li)
print(li)



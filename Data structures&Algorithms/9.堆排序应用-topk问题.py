# 从n个数中间选前k大的数出来,ex:热搜榜,复杂度O(mlogk),运气不好m可以到n这么大(每次列表出来的数都比堆顶大)
import numpy as np
# 先进行一个复制,从大根堆改成小根堆
def sift(li, low, high):
    '''
    :param li:   列表
    :param low:  根节点位置
    :param high: 最尾端的叶子节点位置+1
    :return:     不返回值,把li变成堆(有序)
    '''
    i = low                     # 父节点
    j = i*2 + 1                 # 左孩子节点
    tmp = li[low]               # 把原来的堆顶(省长)记录下来
    while j <= high:            # 还有孩子节点就一直循环
        # 先从孩子节点里挑出最强的来
        if j+1 <= high and li[j+1] < li[j]:
            j = j+1             # j指向右孩子节点
        if li[j] < tmp:         # 下级比领导还强(小),换人
            li[i] = li[j]
            i = j
            j = i*2 + 1
        else:                   # 贬官到此为止
            break
    li[i] = tmp                 # 有两种可能:1.break出来的,tmp就在i这个为止当官正好. 2.j > high了,tmp一蹦到底当村民.
# li = [50, 62, 44]
# sift(li, 0, 3)
# print(li)
def topk(li, k):                        # 步骤: 1.先把列表建成一个体积为k的小根堆  2.遍历列表,如果比堆顶的数小就跳过,如果比堆顶大就替换堆顶,然后进行一次调整
    top_k = li[:k]                      # 先把前k个元素定为top_k
    for i in range((k-2)//2, -1, -1):
        # 倒着来,n-2 = (n-1)-1,n-1是最后一个叶子的下标,再减1是从孩子节点推父节点的公式
        sift(top_k, i, k-1)             # 建堆
    # print(top_k)
    for i in range(k, len(li)):         # 拿列表中剩下的元素和堆顶元素(门槛)比较
        if li[i] > top_k[0]:            # 每次替换元素就调整一次
            top_k[0] = li[i]
            sift(top_k, 0, k-1)
    print('生成前5名小根堆: ', top_k)
    # 选出了topk,接下来从大到小排序
    for i in range(k-1, -1, -1):        # 倒序弹出
        top_k[0], top_k[i] = top_k[i], top_k[0]
        sift(top_k, 0, i-1)
    return top_k

li = [np.random.randint(0, 100) for _ in range(20)]
print('生成列表: ', li)
print('排序前5名: ', topk(li, 5))

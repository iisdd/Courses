# 有点垃圾,不关键,遍历元素加插入排序,时间复杂度可以到O(n^2)
def bucket_sort(li, n=100, max_num=10000):                  # 需要输入桶的数量,最大的数字
    buckets = [[] for _ in range(n)]                        # 创建桶
    for val in li:                                          # 遍历列表,把元素丢进相应的桶里
        idx = min(val//(max_num//n), n-1)                   # 不让桶的下标超上限,超过的都丢进最后一个桶
        buckets[idx].append(val)
        for j in range(len(buckets[idx])-1, 0, -1):         # 用插入排序保持桶内有序,右边的和左边的比,小的话就交换,所以只用比到下标为1的元素
            if buckets[idx][j] < buckets[idx][j-1]:         # 小的牌往前插
                buckets[idx][j], buckets[idx][j-1] = buckets[idx][j-1], buckets[idx][j]
    # 到这里,每个桶都排好序了,现在只需要把桶按顺序拼起来就行了
    li.clear()
    for buc in buckets:
        li.extend(buc)


import random
li = [random.randint(0, 10000) for _ in range(100)]
random.shuffle(li)
print(li)
bucket_sort(li)
print(li)









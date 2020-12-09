# 只能排整数,把数字从低位到高位排序,时间复杂度O(kn) k为循环次数,空间复杂度O(k+n)
# 只有装桶输出,装桶输出,没有桶内排序
def radix_sort(li):
    max_num = max(li)
    iters = len(str(max_num))                   # 循环排序次数
    for iter in range(iters):
        buckets = [[] for _ in range(10)]       # 这一位数的桶
        for num in li:
            idx = num // (10**iter) % 10        # 取第iter位数(从右往左)
            buckets[idx].append(num)
        li.clear()
        for buc in buckets:                     # li按第iter位数排好序了
            li.extend(buc)

import random
li = [random.randint(0, 10000) for _ in range(100)]
random.shuffle(li)
print(li)
radix_sort(li)
print(li)

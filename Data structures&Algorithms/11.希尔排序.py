# 时间复杂度不确定,和选取的序列有关,下至O(n^4/3)上至O(n^2)
# 比堆排序慢(NB三人组垫底)
# 先修改一波插入排序代码,跳着分组的
def insert_sort_gap(li, gap):
    for i in range(gap, len(li)):        # 抽到的牌
        j = i-gap                        # 往左边看有序区
        tmp = li[i]
        while j >= 0 and li[j] > tmp:  # 抽到的牌比有序区的牌小,有序区的牌右移一位,抽到的牌左移
            li[j+gap] = li[j]
            j -= gap
        li[j+gap] = tmp

def shell_sort(li):
    d = len(li)//2
    while d:
        insert_sort_gap(li, d)
        d //= 2

import random
li = list(range(20))
random.shuffle(li)
print(li)
shell_sort(li)
print(li)





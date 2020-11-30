# 复杂度O(n^2)
def insert_sort(li):
    for i in range(1, len(li)):        # 抽到的牌
        j = i-1                        # 往左边看有序区
        tmp = li[i]
        while j >= 0 and li[j] > tmp:  # 抽到的牌比有序区的牌小,有序区的牌右移一位,抽到的牌左移
            li[j+1] = li[j]
            j -= 1
        li[j+1] = tmp

li = [8,9,6,4,3,7]
insert_sort(li)
print(li)

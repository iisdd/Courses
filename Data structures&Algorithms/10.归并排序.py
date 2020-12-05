# 复杂度: O(nlogn),需要额外的空间复杂度O(n)
# 要用到递归,先将一个长列表循环二分成单个元素(单个元素肯定有序),接着调用merge函数将两个有序列表归并
# 先定义一个merge,实现将两个有序列表合成一个有序列表(从小到大)
def merge(li, low, mid, high):                  # low是左列表开头,mid是左列表结尾,mid+1为右列表开头,high为右列表结尾
    i = low                                     # 左列表指针
    j = mid+1                                   # 右列表指针
    li_tmp = []                                 # 用来存合成后的列表
    while i<=mid and j<=high:
        if li[i] < li[j]:
            li_tmp.append(li[i])
            i += 1
        else:
            li_tmp.append(li[j])
            j += 1
    # 到这里有一个列表先走完了,把剩下的补上去
    while i <= mid:
        li_tmp.append(li[i])
        i += 1
    while j <= high:
        li_tmp.append(li[j])
        j += 1
    # 替换原来的li片段
    li[low : high+1] = li_tmp
    return li

li = [1,3,5,7,2,4,6,8]
print(merge(li, 0, 3, 7))

def merge_sort(li, low, high):                  # 用递归进行排序
    if low < high:                              # 至少要有两个元素
        mid = (low + high) // 2
        merge_sort(li, low, mid)                # 对左边进行归并排序,如果low==mid就跳过
        merge_sort(li, mid+1, high)             # 对右边进行归并排序
        merge(li, low, mid, high)               # 对排序好的两个列表进行归并


import random
li = list(range(20))
random.shuffle(li)
print(li)
merge_sort(li, 0, len(li)-1)
print(li)



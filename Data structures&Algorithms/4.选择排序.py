# 复杂度O(n^2),相比冒泡排序,减少了交换的次数
def select_sort_simple(li):         # 每次把最小值挑出来放新列表里
    res = []
    counter = 0                     # 记录遍历的次数
    n = len(li)
    for i in range(n):              # 遍历n遍
        tmp_min = float('inf')
        tmp_idx = 0
        for j in range(n-counter):  # 每次遍历后列表都会变短
            if li[j] < tmp_min:
                tmp_min = li[j]
                tmp_idx = j
        counter += 1
        res.append(li.pop(tmp_idx))
    return res

li = [8,9,6,4,3,7]
print(select_sort_simple(li))

def select_sort_improve(li):        # 原地排序
    counter = 0
    n = len(li)
    for i in range(n-1):            # 遍历n-1遍就够了
        tmp_min = float('inf')
        tmp_idx = 0
        for j in range(counter, n):  # 每次遍历后无序区都会变短
            if li[j] < tmp_min:
                tmp_min = li[j]
                tmp_idx = j
        li[counter], li[tmp_idx] = li[tmp_idx], li[counter]
        counter += 1

    return li

li = [8,9,6,4,3,7]
print(select_sort_improve(li))

def select_sort_answer(li):             # 整体思路: 每次找无序区的最小值放无序区的排头
    for i in range(len(li)-1):          # 优化点1: 只排n-1次,最后那个数自动放最后
        min_loc = i
        for j in range(i+1, len(li)):   # 优化点2: 默认第i个元素为无序区最小值
            if li[j] < li[min_loc]:
                min_loc = j
        li[i], li[min_loc] = li[min_loc], li[i]

li = [8,9,6,4,3,7]
print(select_sort_improve(li))


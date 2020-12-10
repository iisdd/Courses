'''
    给定一个 m*n 的二维列表,查找一个数是否存在。列表有下列性质
    1.每一行的列表从左到右已经排序好了。
    2.每一行的第一个数比上一行最后一个数大
    Ex:
    [
    [1 ,  3,  5,  7],
    [10, 11, 16, 20],
    [23, 30, 34, 50]
    ]
'''
def exsit(li, num):
    # 进行两次二分查找
    m, n = len(li), len(li[0])
    l, r = 0, m-1
    row = -1
    while l < r:
        mid = (l+r) // 2
        if mid == m-1 or (li[mid][0]<=num and li[mid+1][0]>num):        # mid为要找的行
            row = mid
            break
        elif li[mid][0] < num:
            l = mid+1
        else:
            r = mid-1
    if row == -1:                                                       # 因为l == r跳出循环的
        row = l
    print('所在行: ', row)
    # 再来一遍双指针
    l, r = 0, n-1
    column = -1
    tmp = li[row]
    if tmp[-1] < num:                                                   # 一行的最大值都没num大
        return False
    while l <= r:
        mid = (l+r) // 2
        if tmp[mid] == num:
            column = mid
            print('所在列: ', column)
            return True
        elif tmp[mid] < num:
            l = mid + 1
        else:
            r = mid - 1
    return False

li =[
    [1 ,  3,  5,  7],
    [10, 11, 16, 20],
    [23, 30, 34, 50]
    ]
print(exsit(li, 16))






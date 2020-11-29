# 复杂度O(logn)
def binary_search(li, val): # 使用二分法在列表中查找指定元素的index, li为从小到大排序好的列表
    n  = len(li)
    l = 0        # 左指针
    r = n-1      # 右指针
    while l <= r:
        mid = (l+r) // 2
        if li[mid] == val:
            return mid
        elif li[mid] > val:
            r = mid - 1
        else:
            l = mid + 1
    return None

print(binary_search([1,2,3,4,5,6], 6))
print(binary_search([1,2,3,4,5,6], 7))

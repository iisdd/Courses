# 复杂度O(nlogn)
# 整体思路分归位,递归两部分
import numpy as np
def _quick_sort(li, left, right):         # data为待排序列表,left,right定义待排序区域
    if left < right:                      # 有两个及以上元素才需要递归
        mid = partition(li, left, right)  # 1.归位
        _quick_sort(li, left, mid-1)      # 2.递归排两边的序
        _quick_sort(li, mid+1, right)

def partition(li, left, right):          # 排好中间那个元素并返回mid的index,左右横跳排序
    tmp = li[left]                       # 把最左边的数拿出来存着
    while left < right:
        while left < right and li[right] >= tmp:
            # 检查右边的数是不是比tmp大,重合了就跳出
            right -= 1
        li[left] = li[right]             # 把右边小的数字丢进左边的空位,右边指针的位置就成了空位
        while left < right and li[left] <= tmp:
            # 检查左边的数是不是比tmp小
            left += 1
        li[right] = li[left]             # 把左边大的数字丢进右边的空位,左边指针的位置又成了空位
    li[left] = tmp                       # 把最开始左边的数字放回空位,也就是mid的位置上
    return left                          # 返回归位的index

def quick_sort(li):                      # 方便统计运行时间,因为递归会返回很多个时间
    _quick_sort(li, 0, len(li)-1)

# li = [8,9,6,4,3,7]
li = [np.random.randint(0, 100) for _ in range(20)]
quick_sort(li)
print(li)


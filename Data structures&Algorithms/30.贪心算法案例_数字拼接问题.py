# 贪心算法经典案例: 数字拼接问题
# 有一个列表存了很多数字内容的字符串,把它拼接成一个最大的数
# # 把列表元素都转化成字符串
# li = [1,2,3,4]
# li = list(map(str, li))
# print(li)

from functools import cmp_to_key                                # 把python2中的cmp(python3中移除)转化成key函数(sort中的)

li = [32, 94, 128, 1286, 6, 67, 71, 716]

def xy_cmp(x, y):
    if x+y > y+x:
        return 1                                                # 不交换x,y顺序
    elif x+y < y+x:                                             # 交换
        return -1
    else:
        return 0

def number_join(li):
    li = list(map(str, li))
    li.sort(key=cmp_to_key(xy_cmp), reverse=True)
    print(li)
    res = ''
    for num in li:
        res += num
    return res

print(number_join(li))




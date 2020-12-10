'''
    给定一个列表和一个整数,设计算法找到两个数的下标,使得两个数之和为给定的整数。保证结果唯一
'''
def find_idx(li, target):
    dct = {}
    for idx, val in enumerate(li):
        dct[val] = idx
    for key in dct:
        if target - key in dct:
            return [dct[key], dct[target-key]]

li = [1,2,5,4]
target = 3
print(find_idx(li, target))

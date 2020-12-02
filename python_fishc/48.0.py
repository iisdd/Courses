'''
0. 用 while 语句实现与以下 for 语句相同的功能：
for each in range(5):
    print(each)
'''

r = iter(range(5))
while 1:
    try:
        each = next(r)
    except StopIteration:
        break
    print(each)

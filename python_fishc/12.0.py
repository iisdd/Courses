'''
0.你有听说过列表推导式或列表解析吗？
用推导式生成0 到 9 平方的列表
'''
list1 = [i ** 2 for i in range(10)]
print(list1)

'''
问题：请先在 IDLE 中获得下边列表的结果，并按照上方例子把列表推导式还原出来。
>>> list1 = [(x, y) for x in range(10) for y in range(10) if x%2==0 if y%2!=0]
'''
list1 = []
for x in range(10):
    for y in range(10):
        if (x % 2 == 0) and (y % 2 != 0):
            list1.append((x, y))
print(list1)

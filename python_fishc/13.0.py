'''
0.写一个元组生成器（类似于列表推导式）
'''
#   用tuple1.__next__()一个一个显示
tuple1 = (x**2 for x in range(10))
def jixu():
    return tuple1.__next__()
